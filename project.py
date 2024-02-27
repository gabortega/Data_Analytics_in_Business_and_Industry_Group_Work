# Gabriel Ortega
# 2836741O@student.gla.ac.uk

from __future__ import annotations

import random
import string

import numpy
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.cluster as skcluster
import sklearn.preprocessing as skpreproc
from kmodes.kprototypes import KPrototypes
import time
import json
from tqdm import tqdm
from datetime import datetime

from enum import Enum
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from yellowbrick.cluster.elbow import kelbow_visualizer

##############################################################
##############################################################
##############################################################
class Profile:

    def __init__(self, gender: str, age: int, id: str, became_member_on: str, income: int):
        self.id = id
        self.gender = gender
        self.age = age
        self.became_member_on = datetime.strptime(became_member_on, "%Y%m%d")
        self.income = income

    def get_id(self) -> str:
        return self.id

    def get_gender(self) -> str:
        return self.gender

    def get_age(self) -> int:
        return self.age

    def get_became_member_on(self) -> datetime:
        return self.became_member_on

    def get_income(self) -> int:
        return self.income


class ProfileRegistry(list):

    def __init__(self, filename: str):
        super().__init__()
        with open(filename) as file:
            json_data = json.load(file)
            for json_profile in json_data:
                self.append(Profile(
                    id=json_profile["id"],
                    gender=json_profile["gender"],
                    age=json_profile["age"],
                    became_member_on=json_profile["became_member_on"],
                    income=json_profile["income"]
                ))


##############################################################
##############################################################
##############################################################
class Portfolio:

    def __init__(self, id: str, reward: int, channels: list, difficulty: int, duration: int, offer_type: str):
        self.id = id
        self.reward = reward
        self.channels = channels
        self.difficulty = difficulty
        self.duration = duration
        self.offer_type = offer_type

    def get_id(self) -> str:
        return self.id

    def get_reward(self) -> int:
        return self.reward

    def get_channels(self) -> list:
        return self.channels

    def get_difficulty(self) -> int:
        return self.difficulty

    def get_duration(self) -> int:
        return self.duration

    def get_offer_type(self) -> str:
        return self.offer_type


class PortfolioRegistry(list):

    def __init__(self, filename: str):
        super().__init__()
        with open(filename) as file:
            json_data = json.load(file)
            for json_profile in json_data:
                self.append(Portfolio(
                    id=json_profile["id"],
                    reward=json_profile["reward"],
                    channels=json_profile["channels"],
                    difficulty=json_profile["difficulty"],
                    duration=json_profile["duration"],
                    offer_type=json_profile["offer_type"]
                ))


##############################################################
##############################################################
##############################################################
class Transcript:

    def __init__(self, person: str, event: str, value: dict, time: int):
        self.person = person
        self.event = event
        self.value = value
        self.time = time

    def get_person(self) -> str:
        return self.person

    def get_event(self) -> str:
        return self.event

    def get_value(self) -> dict:
        return self.value

    def get_time(self) -> int:
        return self.time


class TranscriptRegistry(list):

    def __init__(self, filename: str):
        super().__init__()
        with open(filename) as file:
            json_data = json.load(file)
            for json_profile in json_data:
                self.append(Transcript(
                    person=json_profile["person"],
                    event=json_profile["event"],
                    value=json_profile["value"],
                    time=json_profile["time"]
                ))


##############################################################
##############################################################
##############################################################
class DataProcessor:

    def __init__(self, profile_filename: str, portfolio_filename: str, transcript_filename: str):
        self.profile_registry = ProfileRegistry(profile_filename)
        self.portfolio_registry = PortfolioRegistry(portfolio_filename)
        self.transcript_registry = TranscriptRegistry(transcript_filename)
        self.filtered_transcripts = []
        self.transactions = np.empty(shape=(0, 4),
                                     dtype=[("profile_id", "U32"), ("time_since_last_transaction", "float"),
                                            ("amount", "float"), ("discount", "float")])
        self.last_transaction_per_id = {}
        self.offers = np.empty(shape=(0, 6),
                               dtype=[("profile_id", "U32"), ("offer_id", "U32"),
                                      ("time_to_offer_view", "float"),
                                      ("time_to_offer_completion", "float")])
        self.last_offer_occurrence_per_id = {profile.get_id(): {} for profile in self.profile_registry}

    def transcript_to_transaction(self, transcript: Transcript) -> numpy.ndarray:
        profile_id = transcript.get_person()
        curr_time = transcript.get_time()
        time_elapsed = curr_time - self.last_transaction_per_id.get(profile_id, curr_time)
        self.last_transaction_per_id[profile_id] = curr_time
        return np.array([(profile_id, time_elapsed, transcript.get_value()["amount"], np.NaN)],
                        dtype=[("profile_id", "U32"), ("time_since_last_transaction", "float"),
                               ("amount", "float"), ("discount", "float")])

    def transcript_to_offer_received(self, transcript: Transcript) -> numpy.ndarray:
        profile_id = transcript.get_person()
        offer_id = transcript.get_value()["offer id"]
        self.last_offer_occurrence_per_id[profile_id][offer_id] = transcript.get_time()
        return np.array([(profile_id, offer_id, np.NaN, np.NaN)],
                        dtype=[("profile_id", "U32"), ("offer_id", "U32"),
                               ("time_to_offer_view", "float"),
                               ("time_to_offer_completion", "float")])

    def get_last_matching_offer(self, profile_id: str, offer_id: str) -> numpy.ndarray:
        return next(filter(lambda entry:
                           entry["profile_id"] == profile_id and entry["offer_id"] == offer_id,
                           self.offers[::-1]))

    def transcript_to_offer_viewed(self, transcript: Transcript) -> None:
        profile_id = transcript.get_person()
        offer_id = transcript.get_value()["offer id"]
        self.get_last_matching_offer(profile_id, offer_id)["time_to_offer_view"] = (
                transcript.get_time() - self.last_offer_occurrence_per_id[profile_id][offer_id])

    def transcript_to_offer_completed(self, transcript: Transcript) -> None:
        self.transactions[-1][3] = np.nan_to_num(self.transactions[-1][3]) + transcript.get_value()["reward"]
        profile_id = transcript.get_person()
        offer_id = transcript.get_value()["offer_id"]
        self.get_last_matching_offer(profile_id, offer_id)["time_to_offer_completion"] = (
                transcript.get_time() - self.last_offer_occurrence_per_id[profile_id][offer_id])

    def run(self) -> None:
        newsletter_offers_ids = [offer.get_id() for offer in self.portfolio_registry if offer.get_reward() == 0]
        for transcript in tqdm(self.transcript_registry):
            if not ("offer id" in transcript.get_value().keys() and
                    transcript.get_value()["offer id"] in newsletter_offers_ids):
                if transcript.get_event() == "transaction":
                    self.transactions = np.append(self.transactions, self.transcript_to_transaction(transcript))
                elif transcript.get_event() == "offer received":
                    self.offers = np.append(self.offers, self.transcript_to_offer_received(transcript))
                elif transcript.get_event() == "offer viewed":
                    self.transcript_to_offer_viewed(transcript)
                elif transcript.get_event() == "offer completed":
                    self.transcript_to_offer_completed(transcript)
        # Save to CSV
        pd.DataFrame(self.transactions).to_csv("transactions_data.csv", index=False)
        pd.DataFrame(self.offers).to_csv("offers_data.csv", index=False)


##############################################################
##############################################################
##############################################################
class DataMergeProcessor:

    def __init__(self, profiles_data_filename: str, transactions_data_filename: str, offers_data_filename: str):
        self.profile_data = pd.read_json(profiles_data_filename)
        self.transactions_data = pd.read_csv(transactions_data_filename)
        self.offers_data = pd.read_csv(offers_data_filename)

    def run(self) -> None:
        # Process transactions data
        processed_transaction_data = self.transactions_data.groupby("profile_id").mean()
        processed_transaction_data.rename(columns={"time_since_last_transaction": "time_since_last_transaction_mean",
                                                   "amount": "amount_mean",
                                                   "discount": "discount_mean"},
                                          inplace=True)
        processed_transaction_data_count = (self.transactions_data
                                            .drop(columns=["time_since_last_transaction", "discount"])
                                            .groupby("profile_id").count())
        processed_transaction_data_count.rename(columns={"amount": "total_transactions"},
                                                inplace=True)
        # processed_transaction_data_max = self.transactions_data.groupby("profile_id").max()
        # processed_transaction_data_min = self.transactions_data.groupby("profile_id").min()
        processed_transaction_data.join(processed_transaction_data_count)
        processed_transaction_data["discount_mean"].fillna(0, inplace=True)

        # Process offers data
        processed_offers_data = (self.offers_data.drop(columns=["offer_id"]).groupby("profile_id").mean())
        processed_offers_data.rename(columns={"time_to_offer_view": "time_to_offer_view_mean",
                                              "time_to_offer_completion": "time_to_offer_completion_mean"},
                                     inplace=True)
        processed_offers_data_count = self.offers_data.groupby("profile_id").count()
        processed_offers_data_count.rename(columns={"offer_id": "offers_received",
                                                    "time_to_offer_view": "offers_viewed",
                                                    "time_to_offer_completion": "offers_completed"},
                                           inplace=True)
        # processed_offers_data_count["offers_expired"] = (
        #         processed_offers_data_count["offers_received"] - processed_offers_data_count["offers_completed"])
        processed_offers_data_count["offers_view_ratio"] = (
                processed_offers_data_count["offers_viewed"].astype(numpy.float32) /
                processed_offers_data_count["offers_received"].astype(numpy.float32))
        processed_offers_data_count["offers_completion_ratio"] = (
                processed_offers_data_count["offers_completed"].astype(numpy.float32) /
                processed_offers_data_count["offers_received"].astype(numpy.float32))
        processed_offers_data = (
            processed_offers_data.join(processed_offers_data_count))

        # Process profiles data
        processed_profiles_data = self.profile_data[["id", "age", "income"]].copy(deep=True)
        processed_profiles_data.rename(columns={"id": "profile_id"}, inplace=True)
        processed_profiles_data.set_index("profile_id", inplace=True)
        processed_profiles_data["age"].replace(118, pd.NA, inplace=True)
        processed_profiles_data["age"].fillna(processed_profiles_data["age"].mean(), inplace=True)
        processed_profiles_data["income"].fillna(processed_profiles_data["income"].mean(), inplace=True)

        # Putting everything together
        data_processed = (
            processed_profiles_data.drop(columns=["age", "income"])
            .join(processed_offers_data.drop(columns=["offers_viewed", "offers_completed"]))
            .join(processed_transaction_data))
        # Handle NA cases
        data_processed.dropna(subset=["amount_mean"], inplace=True)
        (data_processed["time_to_offer_view_mean"]
         .fillna(data_processed["time_to_offer_view_mean"].mean(), inplace=True))
        (data_processed["time_to_offer_completion_mean"]
         .fillna(data_processed["time_to_offer_completion_mean"].mean(), inplace=True))
        (data_processed["offers_received"]
         .fillna(0, inplace=True))
        # (data_processed["offers_viewed"]
        #  .fillna(0, inplace=True))
        # (data_processed["offers_completed"]
        #  .fillna(0, inplace=True))
        # (data_processed["offers_expired"]
        #  .fillna(0, inplace=True))
        (data_processed["offers_view_ratio"]
         .fillna(0.0, inplace=True))
        (data_processed["offers_completion_ratio"]
         .fillna(0.0, inplace=True))
        # Save to CSV
        data_processed.to_csv("processed_data.csv", index=False)


# Process the data and extract data of interest
# dp = DataProcessor("data/profile.json", "data/portfolio.json", "data/transcript.json")
# dp.run()

# Process extracted data and perform clustering
dmp = DataMergeProcessor("data/profile.json", "transactions_data.csv", "offers_data.csv")
dmp.run()

# K-Means cluster & Plot
data_processed = pd.read_csv("processed_data.csv")

# use elbow plot to determine number of clusters
kelbow_visualizer(skcluster.KMeans(), data_processed, k=(1, 14))

kmeans_pred = (skcluster.KMeans(n_clusters=4, random_state=0, n_init="auto")
               .fit_predict(skpreproc.StandardScaler().fit_transform(data_processed)))

x_label = "offers_completion_ratio"
y_label = "offers_received"
plt.scatter(data_processed[x_label], data_processed[y_label], c=kmeans_pred)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title("Plot")
plt.show()

# calculate cluster means and plot bar charts to compare cluster statistics
data_agg = data_processed.copy()
data_agg['cluster'] = kmeans_pred
data_agg = data_agg.groupby('cluster').mean().reset_index()
data_agg = data_agg.melt(id_vars='cluster')

g = sns.FacetGrid(data_agg, col='variable', hue='cluster', col_wrap=5, height=2, sharey=False)
g = g.map(plt.bar, 'cluster', 'value').set_titles("{col_name}")
plt.show()

# count distinct offer types
# frequency
# recency
# gender
# year joined
# total amount (quaterly?)
