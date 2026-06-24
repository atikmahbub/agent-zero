import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// ============================================================
// TOOLS (same calculate + weather from earlier days)
// ============================================================

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

// ============================================================
// PART A — SIMPLE STREAMING (no tools, just to see the mechanic)
// ============================================================

async function simpleStreamingDemo() {
  console.log("\n" + "=".repeat(55));
  console.log("PART A: Simple streaming (no tools)");
  console.log("=".repeat(55) + "\n");

  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Write 2 sentences about Istanbul." }],
    stream: true,
  });

  let chunkCount = 0;
  process.stdout.write("Streaming: ");

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    if (delta) {
      process.stdout.write(delta);
      chunkCount++;
    }
  }

  console.log(`\n\n(Received ${chunkCount} chunks)`);
}

// ============================================================
// PART B — STREAMING WITH TOOL CALLS (the hard part)
// ============================================================

type Message = OpenAI.Chat.ChatCompletionMessageParam;

// Accumulator shape for tool calls being built across chunks
type ToolCallAccumulator = {
  id: string;
  name: string;
  arguments: string;
};

async function runStreamingAgent(userMessage: string) {
  console.log("\n" + "=".repeat(55));
  console.log("PART B: Streaming agent with tool calls");
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const messages: Message[] = [{ role: "user", content: userMessage }];

  let iteration = 0;

  while (true) {
    if (iteration >= 5) break;
    iteration++;
    console.log(`\n--- Iteration ${iteration} ---`);

    const stream = await client.chat.completions.create({
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
      stream: true,
    });

    let accumulatedText = "";
    const toolCallAccumulators: Record<number, ToolCallAccumulator> = {};
    let finishReason: string | null = null;

    process.stdout.write("Streaming response: ");

    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      const delta = choice?.delta;

      // Accumulate plain text content
      if (delta?.content) {
        process.stdout.write(delta.content);
        accumulatedText += delta.content;
      }

      // Accumulate tool call fragments — this is the tricky part
      if (delta?.tool_calls) {
        for (const tcDelta of delta.tool_calls) {
          const index = tcDelta.index;

          if (!toolCallAccumulators[index]) {
            toolCallAccumulators[index] = { id: "", name: "", arguments: "" };
          }

          if (tcDelta.id) toolCallAccumulators[index].id = tcDelta.id;
          if (tcDelta.function?.name)
            toolCallAccumulators[index].name += tcDelta.function.name;
          if (tcDelta.function?.arguments)
            toolCallAccumulators[index].arguments += tcDelta.function.arguments;
        }
      }

      if (choice?.finish_reason) finishReason = choice.finish_reason;
    }

    console.log(); // newline after streaming

    // Done — no tool calls, final answer streamed already
    if (finishReason === "stop") {
      messages.push({ role: "assistant", content: accumulatedText });
      console.log("\n[Done — final answer above was streamed token by token]");
      return;
    }

    // Tool calls were made — now we have complete accumulated arguments
    const toolCallsArray = Object.values(toolCallAccumulators);
    console.log(
      `\n[Reconstructed ${toolCallsArray.length} tool call(s) from stream]`,
    );

    // Push assistant message with tool calls (OpenAI requires specific shape)
    messages.push({
      role: "assistant",
      content: accumulatedText || null,
      tool_calls: toolCallsArray.map((tc, i) => ({
        id: tc.id,
        type: "function" as const,
        function: { name: tc.name, arguments: tc.arguments },
      })),
    });

    // Execute each reconstructed tool call
    for (const tc of toolCallsArray) {
      console.log(`  → ${tc.name}(${tc.arguments})`);

      let args: Record<string, string> = {};
      try {
        args = JSON.parse(tc.arguments);
      } catch {
        console.log(
          "  Error: could not parse accumulated arguments — stream may have been cut off",
        );
      }

      const result = executeTool(tc.name, args);
      console.log(`     Result: ${result}`);

      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: result,
      });
    }
  }
}

// ============================================================
// RUN
// ============================================================

async function main() {
  await simpleStreamingDemo();
  await runStreamingAgent("What is 144 * 37 and what's the weather in Dhaka?");
}

main();
