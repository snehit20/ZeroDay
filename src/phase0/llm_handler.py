from groq import Groq

class LLMHandler:
    def __init__(self):
        self.client = Groq()

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an AI fairness expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def detect_sensitive(self, columns):
        prompt = f"""
        Given dataset columns:
        {columns}

        Identify sensitive features that may cause bias.
        Return only a Python list.
        """
        return self.ask(prompt)

    def explain_bias(self, metrics):
        prompt = f"""
        These are fairness metrics:
        {metrics}

        Explain:
        - Is there bias?
        - Which group is affected?
        - Why it matters?
        """
        return self.ask(prompt)