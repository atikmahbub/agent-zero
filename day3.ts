import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const MEMORY_FILE = path.join(__dirname, "memory.json");
const NOTE_FILE = path.join(__dirname, "notes.json");

type Message = OpenAI.Chat.ChatCompletionMessageParam;

function loadMemory(): Message[] {
  if (!fs.existsSync(MEMORY_FILE)) {
    return [];
  }
  const data = fs.readFileSync(MEMORY_FILE, "utf-8");
  return JSON.parse(data) as Message[];
}

function saveMemory(messages: Message[]) {
  fs.writeFileSync(
    MEMORY_FILE,
    JSON.stringify(messages.slice(-20), null, 2),
    "utf-8",
  );
}

function loadNotes(): string[] {
  if (!fs.existsSync(NOTE_FILE)) {
    return [];
  }
  const data = fs.readFileSync(NOTE_FILE, "utf-8");
  return JSON.parse(data) as string[];
}

function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  toolName: string,
): Promise<T> {
  const timeout = new Promise<T>((_, reject) =>
    setTimeout(
      () => reject(new Error(`Tool ${toolName} timed out after ${ms} ms`)),
      ms,
    ),
  );
  return Promise.race([promise, timeout]);
}

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "calculate",
      description:
        "A tool to perform mathematical calculations. The input should be a string representing the mathematical expression to be evaluated. For example, '2 + 2' or 'sqrt(16)'. The output will be the result of the calculation.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "The mathematical expression to be evaluated.",
          },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_weather",
      description:
        'A tool to get the current weather for a given location. The input should be a string representing the location (e.g., "New York City"). The output will be a string describing the current weather conditions in that location.',
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The location for which to get the current weather.",
          },
        },
        required: ["location"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "divide",
      description:
        'A tool to perform division of two numbers. The input should be an object containing the dividend and divisor. For example, { "dividend": 10, "divisor": 2 }. The output will be the result of the division.',
      parameters: {
        type: "object",
        properties: {
          numerator: {
            type: "number",
            description: "The number to be divided (the dividend).",
          },
          denominator: {
            type: "number",
            description: "The number by which to divide (the divisor).",
          },
        },
        required: ["numerator", "denominator"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "slow_tool",
      description: "A tool that simulates a slow external API call",
      parameters: {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "The input for the slow API call.",
          },
        },
        required: ["input"],
      },
    },
  },
];

async function executeTool(
  name: string,
  args: Record<string, string | number>,
): Promise<string> {
  if (name === "calculate") {
    try {
      const result = Function(`"use strict"; return (${args.expression})`)();
      if (!isFinite(result)) return "Error: result is not a finite number";
      return String(result);
    } catch (e) {
      return `Error: invalid expression — ${(e as Error).message}`;
    }
  }

  if (name === "get_weather") {
    const fakeData: Record<string, string> = {
      dhaka: "32°C, humid, partly cloudy",
      london: "14°C, overcast, light rain",
      "new york": "22°C, sunny, clear skies",
      bogura: "31°C, sunny, light breeze",
    };
    const key = String(args.location).toLowerCase();
    return (
      fakeData[key] ?? `Error: weather data not available for ${args.location}`
    );
  }

  if (name === "divide") {
    const num = Number(args.numerator);
    const den = Number(args.denominator);
    if (den === 0) return "Error: division by zero is not allowed";
    return String(num / den);
  }

  if (name === "slow_tool") {
    // Simulates a tool that takes 5 seconds — will trigger timeout
    await new Promise((resolve) => setTimeout(resolve, 5000));
    return `Processed: ${args.input}`;
  }

  return "Error: unknown tool";
}

async function safeExecuteTool(
  name: string,
  args: Record<string, string | number>,
): Promise<string> {
  try {
    return await withTimeout(executeTool(name, args), 3000, name);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.log(`Tool error (${name}):`, message);
    return `Error: ${message}`;
  }
}

const MAX_ITERATIONS = 10;

async function runAgent(userMessage: string) {
  console.log("\n=== Agent starting ===");
  console.log("User:", userMessage);
  console.log("======================\n");

  const history = loadMemory();
  console.log(`Loaded ${history.length} messages from memory\n`);
  history.push({ role: "user", content: userMessage });

  const messages: Message[] = [...history];

  let iteration = 0;

  while (true) {
    if (iteration >= MAX_ITERATIONS) {
      console.log("Max iterations reached — stopping agent");
      const fallback =
        "I was unable to complete the task within the allowed number of steps.";
      messages.push({ role: "assistant", content: fallback });
      saveMemory(messages);
      return fallback;
    }

    iteration++;

    console.log(`--- Iteration ${iteration}/${MAX_ITERATIONS} ---`);

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant that can perform calculations, provide weather information, and save notes using the provided tools." +
            loadNotes(),
        },
        ...messages,
      ],
      tools: toolDefinitions,
    });

    const choice = response.choices[0];
    console.log("Stop Reason:", choice.finish_reason);

    if (choice.finish_reason === "stop") {
      const finalText = choice.message.content ?? "";
      console.log("\n=== Final Answer ===");
      console.log(finalText);

      // Save assistant response to memory
      messages.push({ role: "assistant", content: finalText });
      saveMemory(messages);

      return finalText;
    }

    messages.push(choice.message);

    const toolCalls = choice.message.tool_calls ?? [];

    for (const toolCall of toolCalls) {
      if (toolCall.type !== "function") continue;
      const fn = (toolCall as OpenAI.Chat.ChatCompletionMessageFunctionToolCall)
        .function;
      console.log("\n=== Tool Use Detected ===");
      console.log(`\nTool called : ${fn.name}`);
      const args = JSON.parse(fn.arguments) as Record<string, string | number>;
      console.log("Arguments:", args);

      const result = await safeExecuteTool(fn.name, args);
      console.log("Tool Result:", result);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

const input = process.argv[2];
if (!input) {
  console.log("Please provide a message for the agent to process.");
  process.exit(0);
}

runAgent(input);
