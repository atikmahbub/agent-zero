import "dotenv/config";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const MEMORY_FILE = path.join(__dirname, "memory.json");

type Message = OpenAI.Chat.ChatCompletionMessageParam;

function loadMemory(): Message[] {
  if (!fs.existsSync(MEMORY_FILE)) {
    return [];
  }
  const data = fs.readFileSync(MEMORY_FILE, "utf-8");
  return JSON.parse(data) as Message[];
}

function saveMemory(messages: Message[]) {
  const trimmed = messages.slice(-20); // Keep only the last 20 messages
  fs.writeFileSync(MEMORY_FILE, JSON.stringify(trimmed, null, 2), "utf-8");
}

function clearMemory() {
  if (fs.existsSync(MEMORY_FILE)) {
    fs.unlinkSync(MEMORY_FILE);
    console.log("Memory cleared.\n");
  }
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
      name: "save_note",
      description:
        "A tool to save a note. The input should be a string representing the content of the note. The output will be a confirmation message indicating that the note has been saved.",
      parameters: {
        type: "object",
        properties: {
          note: {
            type: "string",
            description:
              "The information to remember e.g. 'User's name is Atik'",
          },
        },
        required: ["note"],
      },
    },
  },
];

const NOTES_FILE = path.join(__dirname, "notes.json");

function executeTool(name: string, args: Record<string, string>): string {
  if (name === "calculate") {
    try {
      const result = Function(`"use strict"; return (${args.expression})`)();
      return String(result);
    } catch {
      return "Error: invalid expression";
    }
  }

  if (name === "get_weather") {
    const fakeData: Record<string, string> = {
      dhaka: "32°C, humid, partly cloudy",
      london: "14°C, overcast, light rain",
      "new york": "22°C, sunny, clear skies",
      bogura: "31°C, sunny, light breeze",
    };
    const key = args.location.toLowerCase();
    return fakeData[key] ?? `Weather data not available for ${args.location}`;
  }

  if (name === "save_note") {
    const notes: string[] = fs.existsSync(NOTES_FILE)
      ? JSON.parse(fs.readFileSync(NOTES_FILE, "utf-8"))
      : [];
    notes.push(args.note);
    fs.writeFileSync(NOTES_FILE, JSON.stringify(notes, null, 2));
    return `Note saved: "${args.note}"`;
  }

  return "Unknown tool";
}

function loadNotes(): string {
  if (!fs.existsSync(NOTES_FILE)) return "";
  const notes: string[] = JSON.parse(fs.readFileSync(NOTES_FILE, "utf-8"));
  if (notes.length === 0) return "";
  return (
    "\n\nThings you remember about the user:\n" +
    notes.map((n) => `- ${n}`).join("\n")
  );
}

async function runAgent(userMessage: string) {
  console.log("\n=== Agent starting ===");
  console.log("User:", userMessage);
  console.log("======================\n");

  const history = loadMemory();
  console.log(`Loaded ${history.length} messages from memory\n`);

  history.push({ role: "user", content: userMessage });

  const messages: Message[] = [...history];

  while (true) {
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
      const args = JSON.parse(fn.arguments) as Record<string, string>;
      console.log("Arguments:", args);

      const result = executeTool(fn.name, args);
      console.log("Tool Result:", result);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

const args = process.argv.slice(2);
if (args[0] === "clear") {
  clearMemory();
  process.exit(0);
}

const userMessage = args[0] ?? "Hello, my name is Atik and I am a developer.";

runAgent(userMessage);
