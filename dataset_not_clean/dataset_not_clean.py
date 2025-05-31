import pandas as pd

books = pd.read_csv('D:\Programming\Mine\AI-DataScience\Book_recommendation\dataset_not_clean\Goodreads_BestBooksEver_1-10000.csv')

books = books.loc[~(books['bookImage'].isna())]
books.drop("recommendations", axis = 1, inplace = True)

# using numpy
import numpy as np
books["bookPages"] = np.where(books["bookPages"].isna(),
                              "Unknown",
                              books["bookPages"])


books["bookISBN"] = books["bookISBN"].apply(lambda x: str(x)[:-3] if pd.notna(x) else x)


#using random
import random
books['bookISBN'] = books['bookISBN'].apply(lambda x: "978" + "".join(str(random.randint(0, 9))
                                            for _ in range(9)) if pd.isna(x) else x)
books['bookGenres'] = books['bookGenres'].str.replace(r'/\d+,\d+', '', regex=True).str.replace(r'/\d+', '', regex=True).str.split("|").str[0]


# using api key
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv("D:\Programming\Mine\AI-DataScience\Book_recommendation\API Key.env")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def convert_genres_category(x):
    system_prompt = """
            You are an e-commerce platform system assistant.
            Your main task is to classify book genres as accurately as possible,
            returning only one of the following categories: Fiction, Nonfiction, Love, Literature, Wars, Science, Documentary.
        """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x}
        ]
        )

    return response.choices[0].message.content


books["bookGenres"] = books["bookGenres"].apply(
    lambda x: convert_genres_category(x) if pd.notna(x) else x
)


# save to csv
books.to_csv("D:\Programming\Mine\AI-DataScience\Book_recommendation\first_clean_csv\cleaned.csv", index = False)