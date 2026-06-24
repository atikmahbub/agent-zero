import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ============================================================
// KNOWLEDGE BASE (from Day 6/7)
// ============================================================

const documents = [
  {
    id: "1",
    content:
      "Hair transplant procedures cost between $2000 and $5000 USD. FUE starts at $2000, DHI starts at $3000.",
  },
  {
    id: "2",
    content:
      "Recovery after hair transplant takes 7 to 10 days. Avoid sunlight, swimming, strenuous exercise for 2 weeks.",
  },
  {
    id: "3",
    content:
      "FUE (Follicular Unit Extraction) is minimally invasive. Best for patients under 40 with early to moderate hair loss. Minimal scarring.",
  },
  {
    id: "4",
    content:
      "DHI (Direct Hair Implantation) uses a pen tool for precise placement. Better for density and natural hairline. Slightly higher cost than FUE.",
  },
  {
    id: "5",
    content:
      "FUE is recommended for patients under 40 with early to moderate hair loss. DHI recommended for higher density or precise hairline restoration.",
  },
  {
    id: "6",
    content:
      "Patients over 45 need detailed consultation first as hair loss progression may affect long-term results.",
  },
  {
    id: "7",
    content:
      "Free consultation available for all new patients. English and Arabic speaking staff. Located in Sisli, Istanbul.",
  },
  {
    id: "8",
    content:
      "Payment: credit card, bank transfer, cash in USD/EUR/GBP. No crypto. Payment plans available.",
  },
];

type EmbeddedDoc = {
  id: string;
  content: string;
  embedding: number[];
};

let vectorStore: EmbeddedDoc[] = [];

async function embed(text: string): Promise<number[]> {
  const res = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });

  return res.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (magA * magB);
}

async function indexDocuments() {
  process.stdout.write("Indexing");
  for (const doc of documents) {
    vectorStore.push({ ...doc, embedding: await embed(doc.content) });
    process.stdout.write(".");
  }
  console.log(" done\n");
}

async function semanticSearch(query: string, topN = 3): Promise<string> {
  const qEmbed = await embed(query);

  return vectorStore
    .map((d) => ({
      content: d.content,
      score: cosineSimilarity(qEmbed, d.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN)
    .map((r) => r.content)
    .join("\n\n");
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

async function researchAgent(query: string): Promise<string> {
  console.log(`\n [Research Agent] Query: ${query}`);
  const context = await semanticSearch(query, 3);

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a medical research specialist for a hair transplant clinic.
Answer questions using ONLY the provided context. Be factual and concise.
If the context doesn't contain the answer, say "Information not available."`,
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${query}`,
      },
    ],
  });
  const result = response.choices[0].message.content ?? "";
  console.log(`  [Research Agent] Result: ${result.slice(0, 100)}...`);
  return result;
}

async function financeAgent(query: string): Promise<string> {
  console.log(`\n  [Finance Agent] Query: "${query}"`);

  const tools: OpenAI.Chat.ChatCompletionTool[] = [
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
  ];

  const messages: Message[] = [
    {
      role: "system",
      content: `You are a financial calculator for a hair transplant clinic.
Use the calculate tool for all arithmetic. 
Provide clear cost breakdowns with numbers.`,
    },
    { role: "user", content: query },
  ];

  while (true) {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      tools,
    });

    const choice = response.choices[0];

    if (choice.finish_reason === "stop") {
      const result = choice.message.content ?? "";
      console.log(`  [Finance Agent] Result: ${result.slice(0, 100)}...`);
      return result;
    }

    messages.push(choice.message);

    for (const toolCall of choice.message.tool_calls ?? []) {
      if (!toolCall || toolCall.type !== "function") continue;
      const args = JSON.parse(toolCall.function.arguments) as Record<
        string,
        string
      >;
      const calcResult = Function(
        `"use strict"; return (${args.expression})`,
      )();
      console.log(
        `  [Finance Agent] calculate(${args.expression}) = ${calcResult}`,
      );
      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: String(calcResult),
      });
    }
  }
}

async function eligibilityAgent(
  age: number,
  budgetUsd: number,
  procedure: string,
): Promise<string> {
  console.log(
    `\n  [Eligibility Agent] age=${age}, budget=$${budgetUsd}, procedure=${procedure}`,
  );
  const proc = procedure.toUpperCase();
  const minCost = proc === "DHI" ? 3000 : 2000;

  let result = "";

  if (budgetUsd < minCost) {
    result = `NOT ELIGIBLE for ${proc}: Budget $${budgetUsd} is below minimum $${minCost}`;
  } else if (age > 45) {
    result = `CONDITIONALLY ELIGIBLE for ${proc}: Age ${age} requires consultation first`;
  } else if (age < 18) {
    result = `NOT ELIGIBLE for ${proc}: Must be 18 or older`;
  } else {
    result = `ELIGIBLE for ${proc}: Age ${age} and budget $${budgetUsd} meet all requirements`;
  }

  console.log(`  [Eligibility Agent] Result: ${result}`);
  return result;
}

const orchestratorTools: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "call_research_agent",
      description:
        "Calls the research specialist to find information about procedures, recovery, clinic details. Use for any knowledge questions.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The research question to answer",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "call_finance_agent",
      description:
        "Calls the finance specialist for cost calculations, budget analysis, payment breakdowns.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The financial question or calculation needed",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "call_eligibility_agent",
      description:
        "Calls the eligibility specialist to check if a patient qualifies for a procedure.",
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

async function runOrchestrator(userMessage: string) {
  console.log("\n" + "=".repeat(55));
  console.log("ORCHESTRATOR");
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const messages: Message[] = [{ role: "user", content: userMessage }];

  let iteration = 0;

  while (true) {
    if (iteration >= 10) {
      break;
    }
    iteration++;
    console.log(`\n[Orchestrator] Iteration ${iteration}`);

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an orchestrator for a hair transplant clinic.
You have 3 specialist agents available. Delegate tasks appropriately.

RULES:
- Use call_research_agent for any procedure/clinic/recovery questions
- Use call_finance_agent for any cost calculations or budget questions
- Use call_eligibility_agent when patient age and budget are provided
- Call multiple agents in parallel when tasks are independent
- Synthesize all agent results into one clear final answer`,
        },
        ...messages,
      ],
      tools: orchestratorTools,
    });

    const choice = response.choices[0];

    if (choice.finish_reason === "stop") {
      console.log("\n" + "=".repeat(55));
      console.log("FINAL ANSWER:");
      console.log(choice.message.content);
      return;
    }
    messages.push(choice.message);
    const toolCalls = choice.message.tool_calls ?? [];
    console.log(`[Orchestrator] Delegating to ${toolCalls.length} agent(s)`);

    // Run all agent calls — parallel where possible
    const agentResults = (
      await Promise.all(
        toolCalls.map(async (toolCall) => {
          if (!toolCall || toolCall.type !== "function") return undefined;
          const fn = toolCall.function;
          const args = JSON.parse(fn.arguments) as Record<
            string,
            string | number
          >;
          let result = "";

          if (fn.name === "call_research_agent") {
            result = await researchAgent(String(args.query));
          } else if (fn.name === "call_finance_agent") {
            result = await financeAgent(String(args.query));
          } else if (fn.name === "call_eligibility_agent") {
            result = await eligibilityAgent(
              Number(args.age),
              Number(args.budget_usd),
              String(args.procedure),
            );
          }

          return { tool_call_id: toolCall.id, result };
        }),
      )
    ).filter(
      (result): result is { tool_call_id: string; result: string } =>
        result !== undefined,
    );

    for (const { tool_call_id, result } of agentResults) {
      messages.push({
        role: "tool",
        tool_call_id,
        content: result,
      });
    }
  }
}

async function main() {
  await indexDocuments();

  const query =
    process.argv[2] ??
    "I'm 35 years old with a $3000 budget. Compare FUE and DHI, check if I'm eligible for both, and if my budget allows DHI tell me how much I'd have left after paying the minimum cost.";

  await runOrchestrator(query);
}

main();
