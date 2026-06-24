import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// ============================================================
// KNOWLEDGE BASE — simulating your company docs
// ============================================================
const documents = [
  {
    id: "1",
    content:
      "Hair transplant procedures cost between $2000 and $5000 USD depending on the number of grafts. We offer FUE and DHI techniques.",
  },
  {
    id: "2",
    content:
      "Recovery after hair transplant typically takes 7 to 10 days. Patients should avoid direct sunlight, swimming, and strenuous exercise for 2 weeks.",
  },
  {
    id: "3",
    content:
      "We offer a free consultation for all new patients. Consultations can be booked online or by calling our Istanbul clinic.",
  },
  {
    id: "4",
    content:
      "Our refund policy allows full refunds if cancelled more than 48 hours before the procedure. Cancellations within 48 hours are non-refundable.",
  },
  {
    id: "5",
    content:
      "FUE (Follicular Unit Extraction) is minimally invasive — individual follicles extracted and transplanted. Best for patients wanting minimal scarring. Recovery: 7-10 days.",
  },
  {
    id: "6",
    content:
      "DHI (Direct Hair Implantation) uses a special pen tool for precise placement. Better for density and natural hairline. Slightly higher cost than FUE. Recovery: 7-10 days.",
  },
  {
    id: "7",
    content:
      "Post-operative care: medicated shampoo, antibiotic tablets for 5 days, follow-up video calls at 1 month and 6 months.",
  },
  {
    id: "8",
    content:
      "Payment methods: credit card, bank transfer, cash in USD, EUR, GBP. No cryptocurrency. Payment plans available on request.",
  },
  {
    id: "9",
    content:
      "FUE is recommended for patients under 40 with early to moderate hair loss. DHI is recommended for patients needing higher density or precise hairline restoration.",
  },
  {
    id: "10",
    content:
      "Patients over 45 are advised to have a detailed consultation first as hair loss progression may affect long-term results of transplant procedures.",
  },
];

type EmbeddedDocument = {
  id: string;
  content: string;
  embedding: number[];
};

let vectorStore: EmbeddedDocument[] = [];

async function embed(text: string): Promise<number[]> {
  const response = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  if (magnitudeA === 0 || magnitudeB === 0) return 0; // Avoid division by zero
  return dotProduct / (magnitudeA * magnitudeB);
}

async function indexDocuments() {
  console.log("Indexing documents...");

  for (const doc of documents) {
    const embedding = await embed(doc.content);
    vectorStore.push({ ...doc, embedding });
    process.stdout.write(".");
  }
  console.log("\nIndexing complete.");
}

async function searchKnowledgeBase(query: string, topN = 2): Promise<string> {
  const queryEmbedding = await embed(query);
  const scored = vectorStore
    .map((doc) => ({
      content: doc.content,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
  return scored.map((r) => r.content).join("\n\n");
}

// ============================================================
// TOOLS
// ============================================================

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_knowledge_base",
      description:
        "Search the clinic knowledge base for information about procedures, pricing, recovery, policies.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query e.g. 'FUE recovery time'",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluates a math expression. Use for any calculations.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "Math expression like '3000 * 0.15'",
          },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "check_eligibility",
      description:
        "Checks if a patient is eligible for a specific procedure based on age and budget.",
      parameters: {
        type: "object",
        properties: {
          age: { type: "number", description: "Patient age" },
          budget_usd: { type: "number", description: "Patient budget in USD" },
          procedure: {
            type: "string",
            description: "Procedure name: FUE or DHI",
          },
        },
        required: ["age", "budget_usd", "procedure"],
      },
    },
  },
];

type Args = Record<string, string | number>;

function executeTool(name: string, args: Args): string {
  if (name === "calculate") {
    try {
      const result = Function(`"use strict"; return (${args.expression})`)();
      return String(result);
    } catch {
      return "Error: invalid expression";
    }
  }

  if (name === "check_eligibility") {
    const age = Number(args.age);
    const budget = Number(args.budget_usd);
    const procedure = String(args.procedure).toUpperCase();

    const minCost = procedure === "DHI" ? 3000 : 2000;
    const maxCost = 5000;

    if (budget < minCost) {
      return `Not eligible: budget $${budget} is below minimum cost $${minCost} for ${procedure}`;
    }
    if (age > 45) {
      return `Conditionally eligible: age ${age} requires detailed consultation first before ${procedure}`;
    }
    return `Eligible for ${procedure}: age ${age} and budget $${budget} meet requirements. Estimated cost: $${minCost}–$${maxCost}`;
  }

  return "Error: unknown tool";
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

async function runWithCoT(userMessage: string) {
  console.log("\n" + "=".repeat(55));
  console.log("STRATEGY 1: Chain of Thought");
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const messages: Message[] = [{ role: "user", content: userMessage }];

  let iteration = 0;

  while (true) {
    if (iteration >= 10) {
      console.log("Max iterations reached, stopping agent.");
      break;
    }

    iteration++;

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a hair transplant clinic assistant.

PLANNING RULE — before doing anything, always start your response with:
PLAN:
1. [first step]
2. [second step]
...

Then execute the plan step by step using tools.
Always use search_knowledge_base for clinic information.
Always use check_eligibility when patient age and budget are mentioned.`,
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

    // Show thinking if present
    if (choice.message.content) {
      console.log("\nAgent thinking:");
      console.log(choice.message.content);
    }

    messages.push(choice.message);

    const toolCalls = choice.message.tool_calls ?? [];
    for (const toolCall of toolCalls) {
      if (!toolCall.type || toolCall.type !== "function") continue;
      const fn = toolCall.function;
      const args = JSON.parse(fn.arguments) as Args;
      console.log(`\n  → ${fn.name}(${JSON.stringify(args)})`);

      const result =
        fn.name === "search_knowledge_base"
          ? await searchKnowledgeBase(String(args.query))
          : executeTool(fn.name, args);

      console.log(`     ${result.slice(0, 100)}...`);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

type Plan = {
  steps: Array<{
    step: number;
    description: string;
    tool: string;
    args: Args;
  }>;
};

async function createPlan(userMessage: string): Promise<Plan> {
  console.log("executing plan....");

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a planning agent. Break down the user's request into clear, fact-gathering steps.

Available tools and their args:
- search_knowledge_base: { "query": string }  — look up clinic info
- calculate: { "expression": string }  — a NUMERIC math expression only, e.g. "3000 * 0.15". Never pass words.
- check_eligibility: { "age": number, "budget_usd": number, "procedure": "FUE" | "DHI" }

IMPORTANT:
- Only create steps that map to one of the tools above.
- Do NOT create steps for comparisons, recommendations, or advice — those are produced later from the gathered facts. Never assign such a step to "calculate".
- Each step's "args" must match exactly the args of its chosen tool.

Respond ONLY with valid JSON in this exact format, no markdown, no extra text:
{
  "steps": [
    { "step": 1, "description": "what this step does", "tool": "tool_name", "args": { "...": "..." } }
  ]
}`,
      },
      { role: "user", content: userMessage },
    ],
  });

  const raw = response.choices[0].message.content ?? "{}";

  try {
    return JSON.parse(raw) as Plan;
  } catch {
    console.log("Plan parsing failed, raw:", raw);
    return { steps: [] };
  }
}

async function executePlan(plan: Plan): Promise<string[]> {
  const results: string[] = [];

  for (const step of plan.steps) {
    console.log(`\nStep ${step.step}: ${step.description}`);
    console.log(`  Tool: ${step.tool}(${JSON.stringify(step.args)})`);

    let result = "";
    if (step.tool === "search_knowledge_base") {
      result = await searchKnowledgeBase(String(step.args.query));
    } else if (step.tool === "calculate") {
      result = executeTool("calculate", step.args);
    } else if (step.tool === "check_eligibility") {
      result = executeTool("check_eligibility", step.args);
    } else {
      result = "Unknown tool";
    }

    console.log(`  Result: ${result.slice(0, 100)}...`);
    results.push(`Step ${step.step} (${step.description}): ${result}`);
  }

  return results;
}

async function runWithPlanAndExecute(userMessage: string) {
  console.log("\n" + "=".repeat(55));
  console.log("STRATEGY 2: Plan and Execute");
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const plan = await createPlan(userMessage);
  console.log("\nPlan created:");
  plan.steps.forEach((s) => console.log(`  ${s.step}. ${s.description}`));

  console.log("\nExecuting plan...");
  const results = await executePlan(plan);

  // Phase 3 — Synthesize into final answer
  console.log("\nSynthesizing final answer...");

  const synthesis = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content:
          "You are a helpful clinic assistant. Synthesize the research results into a clear, helpful answer for the patient.",
      },
      {
        role: "user",
        content: `Original question: ${userMessage}\n\nResearch results:\n${results.join("\n\n")}`,
      },
    ],
  });

  console.log("\nFinal Answer:");
  console.log(synthesis.choices[0].message.content);
}

async function main() {
  await indexDocuments();

  const complexQuestion =
    "I'm 35 years old with a budget of $3000. Compare FUE and DHI for me, check if I'm eligible for either, and give me a recommendation.";

  const strategy = process.argv[2] ?? "all";

  if (strategy === "1" || strategy === "all") {
    await runWithCoT(complexQuestion);
  }

  if (strategy === "2" || strategy === "all") {
    await runWithPlanAndExecute(complexQuestion);
  }
}

main();
