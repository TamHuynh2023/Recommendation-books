import pandas as pd

books = pd.read_csv("D:\Programming\Mine\AI-DataScience\Book_recommendation\first_clean_csv\cleaned.csv")

books = books[~books["bookDesc"].isna()]


# check language
from langdetect import detect, DetectorFactory

def is_english(text):
    if detect(text) == 'en':
        return text
    else:
        return "not english"

books["bookDesc"] = books["bookDesc"].apply(lambda x: is_english(x) if pd.notna(x) else x)

books.drop_duplicates(subset="bookDesc", inplace=True)


# delete noncategory
def delete_noncategory(x):
    if x in ["Fiction", "Nonfiction", "Love", "Literature", "Wars", "Science", "Documentary"]:
        return x
    else:
        return "Unknown"

books["bookGenres"] = books["bookGenres"].apply(lambda x: delete_noncategory(x) if pd.notna(x) else x)
books = books.loc[~(books["bookGenres"].isin(["Unknown"]))]


# make category
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv("D:\Programming\Mine\AI-DataScience\Book_recommendation\API Key.env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key)

def make_category(x):
    system_prompt = """
        you are a helpful assistant that can generate text based on existing information,
        and return one of the following results: Fiction, Nonfiction, Love, Literature, Wars, Science, Documentary.
        => Returns only a single word.
    """
    reponse = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x}
        ]
    )
    return reponse.choices[0].message.content


books.loc[books["bookGenres"].isna(), "bookGenres"] = books.loc[books["bookGenres"].isna()].apply(
    lambda x: make_category(x["bookDesc"]), axis=1
)

books["isbn_description"] = books[["bookISBN","bookDesc"]].apply(
    lambda row: str(row["bookISBN"]) + " " + row["bookDesc"], axis=1
)

# make year published
def make_yearpublished(x):
    system_prompt = """
        you are a helpful assistant,
        you are a bookworm you know the release dates of books. And returns the year.
        -> Returns only a single word.
    """

    reponse = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x}
        ]
    )
    return reponse.choices[0].message.content

books["year_published"] = books["bookDesc"].apply(
    lambda x: make_yearpublished(x) if pd.notna(x) else x)


books.to_csv("D:\Programming\Mine\AI-DataScience\Book_recommendation\second_clean_csv\second_cleaned.csv", index = False)
