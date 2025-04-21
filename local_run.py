from local_analyst import *
import time

if __name__ == "__main__":
    api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
    FOREX_OF_INTEREST = "EURUSD"
    event = input("Enter the event description: ")
    general_news = get_fmp_news(api_key)
    forex_news = get_forex_news(api_key, FOREX_OF_INTEREST)

    question1 = f"Imagine you are an economist doing real-time event based analysis. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a macroeconomic perspective? Please provide a detailed analysis."
    question2 = f"Imagine you are a forex trader doing real-time event based trading. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a trading perspective? Please provide a detailed analysis."

    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    llama3_model_path = "/home/junhong/868proj/Meta-Llama-3-8B-Instruct"

    shared_llm = LocalLLM(model_path=llama3_model_path, device="cuda")

    general_analyst = News_Analyst(
        embed_model_name=embed_model_name,
        llm=shared_llm,
        device="cuda",
        chunk_size=512,
        chunk_overlap=0,
        verbose=False
    )

    forex_analyst = News_Analyst(
        embed_model_name=embed_model_name,
        llm=shared_llm,
        device="cuda",
        chunk_size=128,
        chunk_overlap=0,
        verbose=False
    )

    prompt_q1 = f"### Question:\n{question1}\n\n### Answer:\n"
    prompt_q2 = f"### Question:\n{question2}\n\n### Answer:\n"

    time_question_received = time.time()
    general_response = general_analyst.analyze(general_news, prompt_q1)
    print("-" * 80)
    print("General analyst's response:")
    print(general_response)

    forex_response = forex_analyst.analyze(forex_news, prompt_q2)
    print("-" * 80)
    print("Forex analyst's response:")
    print(forex_response)

    manager = decision_maker(llm=shared_llm)
    prompt = f"Imagine you are a manager of a hedge fund and here are the two analyses from your analysts:\n\n{general_response}\n\n{forex_response}\n\nBased on these two analyses, what is your long/short decision? Please provide a detailed but short answer."
    prompt_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
    decision = manager.make_decision(prompt_prompt)
    print("-" * 80)
    time_last_token = time.time()
    print(f"Decision: {decision}")
    print(f"Total time taken: {time_last_token - time_question_received:.2f} seconds")