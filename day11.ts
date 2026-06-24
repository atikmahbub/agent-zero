import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

type Span = {
  name: string;
  type: "llm_call" | "tool_execution";
  startTime: number;
  endTime?: number;
  durationMs?: number;
  metadata: Record<string, unknown>;
  status: "running" | "success" | "error";
};

type Trace = {
  traceId: string;
  userMessage: string;
  startTime: number;
  endTime?: number;
  durationMs?: number;
  spans: Span[];
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUsd: number;
  status: "running" | "success" | "error" | "max_iterations";
};

const PRICE_PER_M_INPUT = 0.15;
const PRICE_PER_M_OUTPUT = 0.6;

function createTrace(userMessage: string): Trace {
  return {
    traceId: `trace_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    userMessage,
    startTime: Date.now(),
    spans: [],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalCostUsd: 0,
    status: "running",
  };
}

function startSpan(
  trace: Trace,
  name: string,
  type: Span["type"],
  metadata: Record<string, unknown> = {},
): Span {
  const span: Span = {
    name,
    type,
    startTime: Date.now(),
    metadata,
    status: "running",
  };
  trace.spans.push(span);
  return span;
}

function endSpan(
  span: Span,
  status: "success" | "error",
  extraMetadata: Record<string, unknown> = {},
) {
  span.endTime = Date.now();
  span.durationMs = span.endTime - span.startTime;
  span.status = status;
  span.metadata = { ...span.metadata, ...extraMetadata };
}

function endTrace(trace: Trace, status: Trace["status"]) {
  trace.endTime = Date.now();
  trace.durationMs = trace.endTime - trace.startTime;
  trace.status = status;
}

function calculateCost(inputTokens: number, outputTokens: number): number {
  return (
    (inputTokens / 1_000_000) * PRICE_PER_M_INPUT +
    (outputTokens / 1_000_000) * PRICE_PER_M_OUTPUT
  );
}

function printTrace(trace: Trace) {
  console.log("\n" + "=".repeat(60));
  console.log(`TRACE: ${trace.traceId}`);
  console.log(`Query: "${trace.userMessage}"`);
  console.log("=".repeat(60));

  for (const span of trace.spans) {
    const icon =
      span.status === "success" ? "✓" : span.status === "error" ? "✗" : "…";
    const indent = "  ";

    if (span.type === "llm_call") {
      console.log(
        `${indent}${icon} [llm_call] ${span.durationMs}ms — ${span.metadata.inputTokens} in / ${span.metadata.outputTokens} out tokens — stop: ${span.metadata.stopReason}`,
      );
    } else {
      console.log(
        `${indent}${icon} [tool_execution] ${span.name} — ${span.durationMs}ms — args: ${JSON.stringify(span.metadata.args)}`,
      );
    }
  }

  console.log("-".repeat(60));
  console.log(`Total duration : ${trace.durationMs}ms`);
  console.log(
    `Total tokens   : ${trace.totalInputTokens} in / ${trace.totalOutputTokens} out`,
  );
  console.log(`Total cost     : $${trace.totalCostUsd.toFixed(6)}`);
  console.log(`Status         : ${trace.status}`);
  console.log("=".repeat(60));
}

function saveTrace(trace: Trace) {
  const dir = path.join(__dirname, "traces");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);
  fs.writeFileSync(
    path.join(dir, `${trace.traceId}.json`),
    JSON.stringify(trace, null, 2),
  );
}

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluates a math expression",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "Math expression" },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Gets current weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"],
      },
    },
  },
];

function executeTool(name: string, args: Record<string, string>): string {
  if (name === "calculate") {
    try {
      return String(Function(`"use strict"; return (${args.expression})`)());
    } catch {
      return "Error: invalid expression";
    }
  }
  if (name === "get_weather") {
    const data: Record<string, string> = {
      dhaka: "32°C, humid, partly cloudy",
      bogura: "31°C, sunny, light breeze",
    };
    return data[args.city?.toLowerCase()] ?? "28°C, clear skies";
  }
  return "Error: unknown tool";
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

async function runTracedAgent(userMessage: string) {
  const trace = createTrace(userMessage);
  const messages: Message[] = [
    {
      role: "user",
      content: userMessage,
    },
  ];

  let iteration = 0;

  try {
    while (true) {
      if (iteration >= 10) {
        endTrace(trace, "max_iterations");
        break;
      }
      iteration++;

      const llmSpan = startSpan(
        trace,
        `llm_call_iteration_${iteration}`,
        "llm_call",
      );

      let response: OpenAI.Chat.ChatCompletion;
      try {
        response = await client.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "system",
              content:
                "You are a helpful assistant. Use tools for math and weather.",
            },
            ...messages,
          ],
          tools: toolDefinitions,
        });
      } catch (error) {
        endSpan(llmSpan, "error", { error: String(error) });
        throw error;
      }

      const choice = response.choices[0];
      const usage = response.usage;

      trace.totalInputTokens += usage?.prompt_tokens ?? 0;
      trace.totalOutputTokens += usage?.completion_tokens ?? 0;
      trace.totalCostUsd += calculateCost(
        usage?.prompt_tokens ?? 0,
        usage?.completion_tokens ?? 0,
      );

      endSpan(llmSpan, "success", {
        inputTokens: usage?.prompt_tokens ?? 0,
        outputTokens: usage?.completion_tokens ?? 0,
        stopReason: choice.finish_reason,
      });

      if (choice.finish_reason === "stop") {
        endTrace(trace, "success");
        printTrace(trace);
        saveTrace(trace);
        console.log("\nFinal Answer:", choice.message.content);
        return;
      }

      messages.push(choice.message);

      for (const toolCall of choice.message.tool_calls ?? []) {
        if (!toolCall || toolCall.type !== "function") continue;
        const fn = toolCall.function;
        const args = JSON.parse(fn.arguments) as Record<string, string>;

        // Span: tool execution
        const toolSpan = startSpan(trace, fn.name, "tool_execution", { args });

        try {
          const result = executeTool(fn.name, args);
          endSpan(toolSpan, "success", { result });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: result,
          });
        } catch (error) {
          endSpan(toolSpan, "error", { error: String(error) });
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: `Error: ${error}`,
          });
        }
      }
    }
  } catch (error) {
    endTrace(trace, "error");
    console.error("Agent error:", error);
  }
  printTrace(trace);
  saveTrace(trace);
}

async function main() {
  await runTracedAgent("What is 144 * 37 and weather in Dhaka?");
  await runTracedAgent("What is 500 * 200 * 3?");
}
main();
