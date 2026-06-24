import "dotenv/config";
import OpenAI from "openai";
import readline from "readline/promises";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "check_eligibility",
      description:
        "Checks if a patient is eligible for a procedure. Safe, read-only.",
      parameters: {
        type: "object",
        properties: {
          age: { type: "number" },
          budget_usd: { type: "number" },
          procedure: { type: "string" },
        },
        required: ["age", "budget_usd", "procedure"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluates a math expression. Safe, read-only.",
      parameters: {
        type: "object",
        properties: { expression: { type: "string" } },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "book_appointment",
      description:
        "Books a procedure appointment for the patient. This is a write operation with real consequences.",
      parameters: {
        type: "object",
        properties: {
          patient_name: { type: "string" },
          procedure: { type: "string" },
          date: { type: "string" },
        },
        required: ["patient_name", "procedure", "date"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "process_refund",
      description:
        "Processes a refund payment to the patient. This is a write operation involving real money.",
      parameters: {
        type: "object",
        properties: {
          patient_name: { type: "string" },
          amount_usd: { type: "number" },
        },
        required: ["patient_name", "amount_usd"],
      },
    },
  },
];

const TOOL_REQUIRING_APPROVAL = new Set(["book_appointment", "process_refund"]);

function requiresApproval(toolName: string): boolean {
  return TOOL_REQUIRING_APPROVAL.has(toolName);
}

function executeTool(
  name: string,
  args: Record<string, string | number>,
): string {
  if (name === "calculate") {
    try {
      return String(Function(`"use strict"; return (${args.expression})`)());
    } catch {
      return "Error: invalid expression";
    }
  }

  if (name === "check_eligibility") {
    const age = Number(args.age);
    const budget = Number(args.budget_usd);
    const proc = String(args.procedure).toUpperCase();
    const minCost = proc === "DHI" ? 3000 : 2000;
    if (budget < minCost) return `NOT ELIGIBLE: budget below $${minCost}`;
    return `ELIGIBLE for ${proc}`;
  }

  if (name === "book_appointment") {
    // In a real system this would write to a database / calendar
    return `Appointment booked: ${args.patient_name} — ${args.procedure} on ${args.date}`;
  }

  if (name === "process_refund") {
    // In a real system this would call a payment processor
    return `Refund processed: $${args.amount_usd} to ${args.patient_name}`;
  }

  return "Error: unknown tool";
}

// ============================================================
// HUMAN APPROVAL GATE
// ============================================================

function describeAction(
  toolName: string,
  args: Record<string, string | number>,
): string {
  if (toolName === "book_appointment") {
    return `Book a ${args.procedure} appointment for ${args.patient_name} on ${args.date}`;
  }
  if (toolName === "process_refund") {
    return `Refund $${args.amount_usd} to ${args.patient_name}`;
  }
  return `${toolName}(${JSON.stringify(args)})`;
}

async function askForApproval(
  toolName: string,
  args: Record<string, string | number>,
): Promise<boolean> {
  console.log("\n" + "!".repeat(55));
  console.log("APPROVAL REQUIRED");
  console.log(`Action: ${describeAction(toolName, args)}`);
  console.log("!".repeat(55));

  const answer = await rl.question("Approve this action? (yes/no): ");
  return answer.trim().toLowerCase().startsWith("y");
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

async function runAgentWithApproval(userMessage: string) {
  console.log("\n" + "=".repeat(55));
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const messages: Message[] = [{ role: "user", content: userMessage }];
  let iteration = 0;

  while (true) {
    if (iteration >= 10) break;
    iteration++;
    console.log(`\n--- Iteration ${iteration} ---`);

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a clinic assistant. You can check eligibility, calculate costs, book appointments, and process refunds.
Some actions require human approval before execution — if a user rejects an action, acknowledge it and ask what they'd like to do instead.`,
        },
        ...messages,
      ],
      tools: toolDefinitions,
    });

    const choice = response.choices[0];

    if (choice.finish_reason === "stop") {
      console.log("\nFinal Answer:");
      console.log(choice.message.content);
      return;
    }

    messages.push(choice.message);

    for (const toolCall of choice.message.tool_calls ?? []) {
      if (!toolCall || toolCall.type !== "function") continue;
      const fn = toolCall.function;
      const args = JSON.parse(fn.arguments) as Record<string, string | number>;

      console.log(
        `\n  Agent wants to call: ${fn.name}(${JSON.stringify(args)})`,
      );

      let result: string;

      if (requiresApproval(fn.name)) {
        const approved = await askForApproval(fn.name, args);
        if (approved) {
          console.log("  → Approved. Executing...");
          result = executeTool(fn.name, args);
        } else {
          console.log("  → Rejected by user.");
          result = `Action rejected by user: ${describeAction(fn.name, args)}. Do not retry this exact action without further instruction.`;
        }
      } else {
        result = executeTool(fn.name, args);
        console.log(`  → Auto-executed (safe): ${result}`);
      }

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

async function main() {
  await runAgentWithApproval(
    "I'm 35 years old, budget $3000. Check if I'm eligible for DHI, and if so book me an appointment for July 15th. My name is Atik.",
  );

  rl.close();
}

main();
