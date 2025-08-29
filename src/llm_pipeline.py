import ollama

class LLMPipeline:
    def __init__(self):
        self.model = "qwen3:1.7b"
        self.system_prompt_en = """
                        You are an AI assistant. Your task is to answer only questions related to the Moon. Response relative with the given context and do not generate long responses. If you see 'No relative documents found' at context that means you must not answer this question. For example reply like 'Please ask questions about the Moon.'
                        Context:
                        {context}
                        """
        self.system_prompt_tr = """
                        Sen bir yapay zeka asistansın.Görevin yalnızca "Ay" (Moon) ile ilgili soruları cevaplamaktır. Sana verilen bağlama dayanarak cevap ver ve uzun cevaplar üretme. Eğer bağlam da 'No relative documents found' görürsen bu soruya kesinlikle cevap vermemelisin. Örneğin 'Lütfen ay ile alakalı sorular sorun.' tarzında cevaplar verebilirsin.
                        Bağlam:
                        {context}
                        """
        self.prompt_not_lang = "You are an assistant that provides information to user about Moon. User asked a question in an unrecognized language. Kindly tell the user 'please ask question in English or Turkish'"
    def __call__(self,query,documents,language):
        if language == "tr":
            chat_template = [
                               {
                                   "role":"system",
                                   "content": self.system_prompt_tr.format(context=documents)
                               },
                               {
                                   "role":"user",
                                   "content":f"Kullanıcı sorusu: {query}"
                               }
                           ]
        elif language == "en":
            chat_template = [
                               {
                                   "role":"system",
                                   "content": self.system_prompt_en.format(context=documents)
                               },
                                {
                                    "role":"user",
                                    "content":f"User's question: {query}"
                                }
                           ]
        else:
            chat_template = [
                               {
                                   "role":"user",
                                   "content": self.prompt_not_lang
                               }
                           ]
        response = ollama.chat(model=self.model,messages=chat_template,think=False)
        return response.message.content