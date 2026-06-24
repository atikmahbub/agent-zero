import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state e.g. San Francisco, CA",
          },
        },
        required: ["location"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "celsius_to_fahrenheit",
      description: "Convert Celsius to Fahrenheit",
      parameters: {
        type: "object",
        properties: {
          celsius: {
            type: "number",
            description: "The temperature in Celsius",
          },
        },
        required: ["celsius"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluates a math expression. Always use this for math.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "Math expression like '144 * 37'",
          },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_exchange_rate",
      description: "Gets the exchange rate between two currencies.",
      parameters: {
        type: "object",
        properties: {
          from: {
            type: "string",
            description: "Source currency code e.g. 'USD'",
          },
          to: {
            type: "string",
            description: "Target currency code e.g. 'BDT'",
          },
        },
        required: ["from", "to"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "convert_currency",
      description:
        "Converts an amount from one currency to another using a given exchange rate.",
      parameters: {
        type: "object",
        properties: {
          amount: {
            type: "number",
            description: "The amount to convert",
          },
          rate: {
            type: "number",
            description: "The exchange rate to apply",
          },
        },
        required: ["amount", "rate"],
      },
    },
  },
];

type Args = Record<string, string | number>;

function executeTool(name: string, args: Args): string {
  if (name === "get_weather") {
    const fakeData: Record<string, string> = {
      dhaka: "32°C, humid, partly cloudy",
      london: "14°C, overcast, light rain",
      newyork: "22°C, sunny, clear skies",
      bogura: "31°C, sunny, light breeze",
      tokyo: "18°C, cool, mostly clear",
    };
    const key = String(args.city).toLowerCase().replace(/\s/g, "");
    return fakeData[key] ?? `28°C, clear skies`;
  }

  if (name === "celsius_to_fahrenheit") {
    const c = Number(args.celsius);
    const f = (c * 9) / 5 + 32;
    return `${f}°F`;
  }

  if (name === "calculate") {
    try {
      const result = Function(`"use strict"; return (${args.expression})`)();
      return String(result);
    } catch {
      return "Error: invalid expression";
    }
  }

  if (name === "get_exchange_rate") {
    // Fake rates
    const rates: Record<string, number> = {
      "USD-BDT": 110,
      "USD-EUR": 0.92,
      "EUR-BDT": 119,
      "BDT-USD": 0.0091,
    };
    const key = `${String(args.from).toUpperCase()}-${String(args.to).toUpperCase()}`;
    const rate = rates[key];
    if (!rate)
      return `Error: exchange rate not available for ${args.from} to ${args.to}`;
    return String(rate);
  }

  if (name === "convert_currency") {
    const result = Number(args.amount) * Number(args.rate);
    return String(result.toFixed(2));
  }

  return "Error: unknown tool";
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

const MAX_ITERATIONS = 10;

async function runAgent(userMessage: string, scenario: string) {
  console.log(`\n${"=".repeat(50)}`);
  console.log(`SCENARIO: ${scenario}`);
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(50));

  const messages: Message[] = [{ role: "user", content: userMessage }];
  let iteration = 0;
  let totalToolCalls = 0;

  while (true) {
    if (iteration >= MAX_ITERATIONS) {
      console.log("Max iterations reached, stopping agent.");
      break;
    }

    iteration++;
    console.log(`\n--- Iteration ${iteration} ---`);

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "You are a helpful assistant. Use tools for all calculations and data lookups. Never compute values yourself.",
        },
        ...messages,
      ],
      tools: toolDefinitions,
    });

    const choice = response.choices[0];
    console.log("Stop reason:", choice.finish_reason);

    if (choice.finish_reason === "stop") {
      const finalText = choice.message.content ?? "";
      console.log("\n--- Final Answer ---");
      console.log(finalText);
      console.log(
        `\nStats: ${iteration} iterations, ${totalToolCalls} total tool calls`,
      );
      return;
    }

    messages.push(choice.message);
    const toolCalls = choice.message.tool_calls ?? [];
    console.log(`Tools called this iteration: ${toolCalls.length}`);
    totalToolCalls += toolCalls.length;

    for (const toolCall of toolCalls) {
      if (toolCall.type !== "function") continue;
      const fn = toolCall.function;
      const args = JSON.parse(fn.arguments) as Args;

      console.log(`  → ${fn.name}(${JSON.stringify(args)})`);
      const result = executeTool(fn.name, args);
      console.log(`     Result: ${result}`);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

async function main() {
  const scenario = process.argv[2] ?? "all";

  if (scenario === "a" || scenario === "all") {
    await runAgent(
      "What is the weather in Dhaka, London, and Tokyo?",
      "A — Pure parallel (3 independent tools)",
    );
  }

  // SCENARIO B — Pure chain: each step depends on the previous
  if (scenario === "b" || scenario === "all") {
    await runAgent(
      "Get the weather in Dhaka, then tell me that temperature in Fahrenheit",
      "B — Chained (weather → celsius_to_fahrenheit)",
    );
  }

  // SCENARIO C — Mixed: some parallel, some chained
  if (scenario === "c" || scenario === "all") {
    await runAgent(
      "Get the weather in Dhaka and London. Also get the USD to BDT exchange rate, then tell me how much 500 USD is in BDT.",
      "C — Mixed (parallel weather + chained currency conversion)",
    );
  }
}

main();
