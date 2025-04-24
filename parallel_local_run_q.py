import time
import torch
from multiprocessing import Process, Queue
from parallel_local_analyst_q import News_Analyst, LocalLLM, decision_maker
from fetch_news import get_fmp_news, get_forex_news


def analysis_worker(device: str, model_path: str, news_data, prompt: str, result_queue: Queue):
    """worker process function"""
    try:
        # explicitly bind device
        torch.cuda.set_device(int(device.split(":")[1]))

        # initialize components for this process
        llm = LocalLLM(model_path=model_path, device=device)
        analyst = News_Analyst(
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm=llm,
            device=device,
            chunk_size=512 if "cuda:0" in device else 128,
            chunk_overlap=0
        )

        # perform analysis
        result = analyst.analyze(news_data, prompt)
        result_queue.put((device, result))
    except Exception as e:
        result_queue.put((device, f"Error: {str(e)}"))


def main():
    api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
    FOREX_OF_INTEREST = "EURUSD"
    model_path = "/home/junhong/868proj/Meta-Llama-3-8B-Instruct"
    event = input("Enter the event description: ")
    devices = ["cuda:2", "cuda:0"]

    analysis_start_time = time.time()
    general_news = get_fmp_news(api_key)
    forex_news = get_forex_news(api_key, "EURUSD")

    prompts = [
        (
            f"Imagine you are an economist doing real-time event based analysis. "
            f"Here is an event: {event}.\nWhat potential impact would this event have on "
            f"{FOREX_OF_INTEREST} exchange rate from a macroeconomic perspective? Please provide a detailed analysis."
        ),
        (
            f"Imagine you are a forex trader doing real-time event based trading. "
            f"Here is an event: {event}.\nWhat potential impact would this event have on "
            f"{FOREX_OF_INTEREST} exchange rate from a trading perspective? Please provide a detailed analysis."
        )
    ]

    result_queue = Queue()

    # start worker processes
    processes = []
    for i, device in enumerate(devices):
        p = Process(
            target=analysis_worker,
            args=(
                device,
                model_path,
                general_news if i == 0 else forex_news,
                prompts[i],
                result_queue
            )
        )
        processes.append(p)
        p.start()

    [p.join() for p in processes]

    # collect results
    results = {}
    while not result_queue.empty():
        device, result = result_queue.get()
        results[device] = result

    analysis_end_time = time.time()
    anaylsis_time_cost = analysis_end_time - analysis_start_time
    print("Analysis time cost:", anaylsis_time_cost)
    print("\n=== Analysis Results ===")
    for device in devices:
        print(f"[{device}] Result:\n{results.get(device, 'No result')}\n")

    # make decision
    decision_device = devices[0]
    torch.cuda.set_device(int(decision_device.split(":")[1]))
    decision_start_time = time.time()
    torch.cuda.set_device(int(decision_device.split(":")[1]))
    manager = decision_maker(LocalLLM(model_path, device=decision_device))
    decision = manager.make_decision(
        f"Economic Analysis:\n{results[devices[0]]}\n\n"
        f"Trading Analysis:\n{results[devices[1]]}\n\n"
        "Based on above, what's your long/short decision? Please give a concise answer."
    )
    decision_end_time = time.time()
    decision_time_cost = decision_end_time - decision_start_time
    print("Decision time cost:", decision_time_cost)
    # print(f"\n=== Final Decision ({decision_device}) ===")
    print(decision)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")