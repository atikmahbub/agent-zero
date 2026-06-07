import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

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
];

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

  return "Unknown tool";
}

const SYSTEM_PROMPT =
  "You are a helpful assistant that can perform calculations and provide weather information using the provided tools.";

async function runAgent(userMessage: string) {
  console.log("Agent Starting...");
  console.log("User:", userMessage);
  console.log("======================\n");

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "user", content: userMessage },
  ];

  while (true) {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
      tools: toolDefinitions,
    });

    const choice = response.choices[0];
    console.log("Stop Reason:", choice.finish_reason);

    messages.push(choice.message);

    if (choice.finish_reason === "stop") {
      const finalText = choice.message.content ?? "";
      console.log("\n=== Final Answer ===");
      console.log("Agent Response:", finalText);
      return finalText;
    }

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

runAgent("What is 144 * 37? Then multiply that result by 2?");
