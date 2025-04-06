from analyst import *
if __name__ == "__main__":
    # Example usage
    api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
    FOREX_OF_INTEREST = "EURUSD"
    event = input("Enter the event description: ")
    general_news = get_fmp_news(api_key) 
    forex_news = get_forex_news(api_key, FOREX_OF_INTEREST)

    

    question1 = f"Imagine you are an economist doing real-time event based analysis. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a macroeconomic perspective? Please provide a detailed analysis."
    question2 = f"image you are a forex trader doing real-time event based trading. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a trading perspective? Please provide a detailed analysis."

    embed_model_name1 = "sentence-transformers/all-MiniLM-L6-v2"
    embed_model_name2 = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name1 = "mistralai/Mistral-7B-Instruct-v0.3"
    llm_model_name2 = "mistralai/Mistral-7B-Instruct-v0.3"
    device = "cuda"  # or "cpu"
    
    # Create instances of the News_Analyst class
    general_analyst = News_Analyst(
        embed_model_name=embed_model_name1, 
        llm_model_name=llm_model_name1, 
        device=device,
        chunk_size=512,
        chunk_overlap=0,
        verbose=False,
        similarity_top_k=10,  
    )
    

    
    forex_analyst = News_Analyst(
        embed_model_name=embed_model_name2, 
        llm_model_name=llm_model_name2, 
        device=device,
        chunk_size=128,
        chunk_overlap=0,
        verbose=False,
        similarity_top_k=10,  
    )
    
    # Analyze the news data
    general_response = general_analyst.analyze(general_news, question1)
    print("-" * 80)
    print("General analyst's reponse:")
    print(general_response)
    forex_response = forex_analyst.analyze(forex_news, question2)
    print("-" * 80)
    print("Forex analyst's reponse:")
    print(forex_response)
    manager = decision_maker("mistralai/Mistral-7B-Instruct-v0.3")
    prompt = f"Image you are a manager of a hedge fund and here are the two analysis from your analysts:\n\n{general_response}\n\n{forex_response}\n\nBased on these two analysis, what is your long/short decision? Please provide a detailed but short answer."
    decision = manager.make_decision(prompt)
    print("-" * 80)
    print(f"Decision: {decision}")
    