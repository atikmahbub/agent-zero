import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// ============================================================
// KNOWLEDGE BASE — simulating your company docs
// ============================================================

const documents = [
  // Istanbul Medic style docs — familiar territory for you
  {
    id: "1",
    content:
      "Hair transplant procedures at our clinic cost between $2000 and $5000 USD depending on the number of grafts required. We offer FUE and DHI techniques.",
  },
  {
    id: "2",
    content:
      "Recovery after hair transplant typically takes 7 to 10 days. Patients should avoid direct sunlight, swimming, and strenuous exercise for 2 weeks after the procedure.",
  },
  {
    id: "3",
    content:
      "We offer a free consultation for all new patients. Consultations can be booked online or by calling our Istanbul clinic directly. We have English and Arabic speaking staff.",
  },
  {
    id: "4",
    content:
      "Our refund policy allows full refunds if cancelled more than 48 hours before the procedure. Cancellations within 48 hours are non-refundable.",
  },
  {
    id: "5",
    content:
      "FUE (Follicular Unit Extraction) is a minimally invasive technique where individual hair follicles are extracted and transplanted. DHI (Direct Hair Implantation) uses a special pen tool for more precise placement.",
  },
  {
    id: "6",
    content:
      "We are located in the Sisli district of Istanbul, Turkey. We provide free airport pickup and hotel accommodation packages for international patients.",
  },
  {
    id: "7",
    content:
      "Post-operative care includes medicated shampoo provided by the clinic, antibiotic tablets for 5 days, and a follow-up video call at 1 month and 6 months after the procedure.",
  },
  {
    id: "8",
    content:
      "Payment methods accepted include credit card, bank transfer, and cash in USD, EUR, or GBP. We do not accept cryptocurrency. Payment plans are available on request.",
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

// Cosine similarity — measures how similar two vectors are
// Returns 1.0 for identical, 0.0 for completely different
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (magA * magB);
}

async function indexDocuments() {
  console.log("Indexing documents...");

  for (const doc of documents) {
    const embedding = await embed(doc.content);
    vectorStore.push({ ...doc, embedding });
    process.stdout.write(".");
  }
  console.log(`\nIndexed ${vectorStore.length} documents\n`);
}

async function search(query: string, topN: number = 3): Promise<string[]> {
  const queryEmbedding = await embed(query);

  const scored = vectorStore.map((doc) => ({
    content: doc.content,
    score: cosineSimilarity(queryEmbedding, doc.embedding),
  }));

  // Sort by similarity score descending
  scored.sort((a, b) => b.score - a.score);

  console.log("\nTop matches:");
  scored.slice(0, topN).forEach((r, i) => {
    console.log(
      `  ${i + 1}. score: ${r.score.toFixed(3)} — ${r.content.slice(0, 60)}...`,
    );
  });

  return scored.slice(0, topN).map((r) => r.content);
}

const toolDefinitions: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_knowledge_base",
      description:
        "Searches the clinic knowledge base for relevant information. Use this for any question about procedures, pricing, location, recovery, refunds, or policies.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "The search query — rephrase the user question as a factual statement for best results e.g. 'hair transplant cost' instead of 'how much does it cost'",
          },
        },
        required: ["query"],
      },
    },
  },
];

async function executeTool(
  name: string,
  args: Record<string, string>,
): Promise<string> {
  if (name === "search_knowledge_base") {
    const results = await search(args.query, 3);
    if (results.length === 0) return "No relevant information found.";
    return results.join("\n\n");
  }
  return "Error: unknown tool";
}

type Message = OpenAI.Chat.ChatCompletionMessageParam;

async function runAgent(userMessage: string) {
  console.log(`\n${"=".repeat(55)}`);
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
    console.log(`\n--- Iteration ${iteration} ---`);

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant for a hair transplant clinic in Istanbul.
          
TOOL RULES:
- Always search the knowledge base before answering any question about the clinic
- Never answer from memory — the knowledge base is the only source of truth
- If the knowledge base doesn't contain the answer, say "I don't have that information — please contact us directly"

BEHAVIOR:
- Be warm and professional
- Keep answers concise and factual
- Always offer to help with follow-up questions`,
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

    const toolCalls = choice.message.tool_calls ?? [];

    for (const toolCall of toolCalls) {
      if (toolCall.type !== "function") continue;
      const fn = toolCall.function;
      const args = JSON.parse(fn.arguments) as Record<string, string>;
      console.log(`\nSearching: "${args.query}"`);

      const result = await executeTool(fn.name, args);

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

async function main() {
  await indexDocuments();

  await runAgent("How much does a hair transplant cost?");
  await runAgent("What should I do after my procedure?");
  await runAgent("Do you accept Bitcoin?");
  await runAgent("Tell me about FUE vs DHI");
  await runAgent("What is your refund policy if I cancel?");
}

main();
