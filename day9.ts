import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

type BlackBoard = {
  facts: Record<string, string | number>;
  log: string[];
};

function createBlackBoard(): BlackBoard {
  return { facts: {}, log: [] };
}

function writeFact(board: BlackBoard, key: string, value: string | number) {
  board.facts[key] = value;
  board.log.push(`[WRITE] ${key} = ${value}`);
  console.log(`  [Blackboard] ${key} = ${value}`);
}

function readFacts(board: BlackBoard): string {
  const entries = Object.entries(board.facts);
  if (entries.length === 0) return "No facts recorded yet.";
  return entries.map(([k, v]) => `${k}: ${v}`).join("\n");
}

const documents = [
  {
    id: "1",
    content:
      "FUE hair transplant starts at $2000 USD. Minimally invasive, individual follicles extracted. Best for under 40, early-moderate hair loss.",
  },
  {
    id: "2",
    content:
      "DHI hair transplant starts at $3000 USD. Uses pen tool for precise placement. Better density, natural hairline.",
  },
  {
    id: "3",
    content:
      "Recovery for both FUE and DHI takes 7-10 days. Avoid sunlight, swimming, exercise for 2 weeks.",
  },
  {
    id: "4",
    content:
      "Patients over 45 need detailed consultation first due to hair loss progression concerns.",
  },
];

type EmbeddedDoc = { id: string; content: string; embedding: number[] };
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

async function semanticSearch(query: string, topN = 2): Promise<string> {
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

async function researchAgent(
  query: string,
  board: BlackBoard,
): Promise<string> {
  console.log(`\n  [Research Agent] Query: "${query}"`);

  const context = await semanticSearch(query, 3);

  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a medical research specialist. Answer using ONLY the provided context.
After answering, extract any numeric cost data and note it clearly like "DHI_COST: 3000".`,
      },
      { role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` },
    ],
  });

  const result = response.choices[0].message.content ?? "";
  console.log(`  [Research Agent] Result: ${result.slice(0, 100)}...`);

  // Write extracted facts to the blackboard
  const fueMatch = result.match(/FUE.*?\$?(\d{3,5})/i);
  const dhiMatch = result.match(/DHI.*?\$?(\d{3,5})/i);
  if (fueMatch) writeFact(board, "fue_cost", Number(fueMatch[1]));
  if (dhiMatch) writeFact(board, "dhi_cost", Number(dhiMatch[1]));

  return result;
}

async function eligibilityAgent(
  age: number,
  budgetUsd: number,
  procedure: string,
  board: BlackBoard,
): Promise<string> {
  console.log(
    `\n  [Eligibility Agent] age=${age}, budget=$${budgetUsd}, procedure=${procedure}`,
  );

  const proc = procedure.toUpperCase();
  // Read cost from blackboard instead of hardcoding
  const minCost =
    (proc === "DHI" ? board.facts["dhi_cost"] : board.facts["fue_cost"]) ??
    (proc === "DHI" ? 3000 : 2000);

  let result = "";
  if (budgetUsd < Number(minCost)) {
    result = `NOT ELIGIBLE for ${proc}: Budget $${budgetUsd} below minimum $${minCost}`;
  } else if (age > 45) {
    result = `CONDITIONALLY ELIGIBLE for ${proc}: Age ${age} requires consultation first`;
  } else {
    result = `ELIGIBLE for ${proc}: meets all requirements`;
  }

  writeFact(board, `${proc.toLowerCase()}_eligibility`, result);
  console.log(`  [Eligibility Agent] Result: ${result}`);
  return result;
}

async function financeAgent(
  budgetUsd: number,
  procedure: string,
  board: BlackBoard,
): Promise<string> {
  console.log(
    `\n  [Finance Agent] Reading blackboard for ${procedure} cost...`,
  );

  const proc = procedure.toUpperCase();
  const cost =
    proc === "DHI" ? board.facts["dhi_cost"] : board.facts["fue_cost"];

  if (cost === undefined) {
    const result = `Cannot calculate — ${proc} cost not yet known on blackboard`;
    console.log(`  [Finance Agent] ${result}`);
    return result;
  }

  const remaining = budgetUsd - Number(cost);
  const result = `Budget $${budgetUsd} - ${proc} cost $${cost} = $${remaining} remaining`;
  writeFact(board, "remaining_budget", remaining);
  console.log(`  [Finance Agent] ${result}`);
  return result;
}

async function runWithBlackBoard(
  userMessage: string,
  age: number,
  budget: number,
  procedure: string,
) {
  console.log("\n" + "=".repeat(55));
  console.log("BLACKBOARD-COORDINATED AGENTS");
  console.log(`User: ${userMessage}`);
  console.log("=".repeat(55));

  const board = createBlackBoard();
  await researchAgent(
    `What is the cost and details of ${procedure} hair transplant?`,
    board,
  );

  await eligibilityAgent(age, budget, procedure, board);
  await financeAgent(budget, procedure, board);

  console.log("\n" + "=".repeat(55));
  console.log("FINAL BLACKBOARD STATE:");
  console.log(readFacts(board));
  console.log("=".repeat(55));

  // Step 3 — Synthesize final answer from blackboard facts
  const synthesis = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content:
          "You are a clinic assistant. Synthesize these facts into a clear answer for the patient.",
      },
      {
        role: "user",
        content: `Patient question: ${userMessage}\n\nFacts gathered:\n${readFacts(board)}`,
      },
    ],
  });

  console.log("\nFINAL ANSWER:");
  console.log(synthesis.choices[0].message.content);
}

async function main() {
  await indexDocuments();

  await runWithBlackBoard(
    "I'm 35 with a $3000 budget, am I eligible for DHI and how much would I have left?",
    35,
    3000,
    "DHI",
  );
}

main();
