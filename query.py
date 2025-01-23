import argparse
import json
import logging
import os
import pickle as pkl
import sqlite3

import mlx.core as mx
from mlx.utils import tree_flatten
from sentence_transformers import SentenceTransformer, util

import utils as lora_utils
from data_gen import table_id2name
from models import LoRALinear

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_wikisql_data(embedding_model: SentenceTransformer) -> tuple[dict, list, list]:
    """
    Loads the wikisql data from the data source or cache, and returns the following:
        db_metadata: a dictionary where the keys are the table IDs and the values are dictionaries with the keys "columns", "types", and "desc".
        corpus: a list of strings, where each string is the description of a table, used as input to the embedding model.
        corpus_embeddings: a tensor with shape (num_tables, embedding_dim) where each row is the embedding of a table description.

    The corpus and corpus_embeddings are cached to disk, so that future calls to this function will load them from disk instead of recomputing them.
    """
    db_metadata: dict[str, dict] = {}
    corpus = []
    corpus_embeddings = None
    if os.path.exists('db_metadata.pkl'):
        logging.info("Loading wikisql data from cache")
        db_metadata = pkl.load(open('db_metadata.pkl', 'rb'))
        corpus = pkl.load(open('db_corpus.pkl', 'rb'))
        corpus_embeddings = pkl.load(open('db_embeddings.pkl', 'rb'))
    else:
        logging.info("Loading wikisql data from source")
        with open('wikisql_data/train.tables.jsonl') as f:
            for line in f:
                table = json.loads(line)
                t_id = table_id2name(table['id'])
                desc = f"table: {t_id}\ncolumns: {', '.join(table['header'])}"
                db_metadata[t_id] = {
                    "columns": table["header"],
                    "types": table["types"],
                    "desc": desc,
                    # "embedding": embedding_model.encode(desc, convert_to_tensor=True),
                }
        pkl.dump(db_metadata, open('db_metadata.pkl', 'wb'))

        corpus = [metadata["desc"] for metadata in db_metadata.values()]
        corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True)

        pkl.dump(corpus, open('db_corpus.pkl', 'wb'))
        pkl.dump(corpus_embeddings, open('db_embeddings.pkl', 'wb'))
    return db_metadata, corpus, corpus_embeddings


def find_relevant_table(question: str, embedding_model: SentenceTransformer, corpus: list, corpus_embeddings) -> str:
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    ans = util.semantic_search(question_embedding, corpus_embeddings, top_k=5)[0]
    # for a in ans:
    #     print(corpus[a['corpus_id']])

    return corpus[ans[0]['corpus_id']]

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The question to ask the model.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    # Generation args
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    # parser.add_argument(
    #     "--data",
    #     type=str,
    #     default="data/",
    #     help="Directory with {train, valid, test}.jsonl files",
    # )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser

def gen_llm_output(prompt_str, model, tokenizer) -> str:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt_str (str): The input prompt.
        model (nn.Module): The model to use for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding and decoding.

    Returns:
        str: The generated text.
    """
    prompt = mx.array(tokenizer.encode(prompt_str))

    tokens = []
    skip = 0
    result = ""

    for token, n in zip(
        lora_utils.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            # print(s[skip:-1], end="", flush=True)
            result += s[skip:-1]
            skip = len(s) - 1
    result += tokenizer.decode(tokens)[skip:]

    if len(tokens) == 0:
        logging.info("No tokens generated for this prompt")

    return result


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    # args.query = "In which year did John Wallace play for Syracuse?"

    # print(find_relevant_table("Tell me what the notes are for South Australia"))

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    db_metadata, corpus, corpus_embeddings = load_wikisql_data(embedding_model)

    logging.info("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(args.model, {})

    # if not os.path.exists('default.npz'):
    #     model.save_weights('default.npz')
    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    logging.info(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6

    logging.info(f"Loading pretrained adapters from {args.adapter_file}")
    model.load_weights(args.adapter_file, strict=False)

    table_id = find_relevant_table(args.query, embedding_model, corpus, corpus_embeddings).split()[1]
    prompt_str = f"{db_metadata[table_id]['desc']}\nQ: {args.query}\nA:"

    print(prompt_str, end="", flush=True)

    sql_query = gen_llm_output(prompt_str, model, tokenizer)

    print(sql_query)
    print("=" * 10)

    conn = sqlite3.connect("wikisql_data/train.db")
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        sql_result = str(cursor.fetchall())
    except sqlite3.OperationalError:
        sql_result = None

    logging.info(sql_result)

    background = f"Database result is {sql_result}" if sql_result else ""
    # model.load_weights('default.npz', strict=False)
    answer = gen_llm_output(f"Answer the following question in a full sentence, as if you were wikipedia. Use the provided data: {args.query}\n{background}.", model, tokenizer)

    print(answer)
