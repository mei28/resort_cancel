import pandas as pd
import numpy as np
import re as re

from base import Feature, get_arguments, generate_features

Feature.dir = 'features'

# """sample usage
# """
# class Pclass(Feature):
#     def create_features(self):
#         self.train['Pclass'] = train['Pclass']
#         self.test['Pclass'] = test['Pclass']


class Hotel(Feature):
    def create_features(self):
        hotels = {"City Hotel": 1, "Resort Hotel": 0}
        self.train['hotel'] = train['hotel'].map(hotels)
        self.test['hotel'] = test['hotel'].map(hotels)


class Lead_time(Feature):
    def create_features(self):
        self.train['lead_time'] = train['lead_time']
        self.test['lead_time'] = test['lead_time']


class Stays_in_weekend_nights(Feature):
    def create_features(self):
        self.train['stays_in_weekend_nights'] = train['stays_in_weekend_nights']
        self.test['stays_in_weekend_nights'] = test['stays_in_weekend_nights']


class Stays_in_week_nights(Feature):
    def create_features(self):
        self.train['stays_in_week_nights'] = train['stays_in_week_nights']
        self.test['stays_in_week_nights'] = test['stays_in_week_nights']


class Stays_in_weekend_nights(Feature):
    def create_features(self):
        self.train['stays_in_weekend_nights'] = train['stays_in_weekend_nights']
        self.test['stays_in_weekend_nights'] = test['stays_in_weekend_nights']


class Adults(Feature):
    def create_features(self):
        self.train['adults'] = train['adults']
        self.test['adults'] = test['adults']


class Children(Feature):
    def create_features(self):
        self.train['children'] = train['children']
        self.test['children'] = test['children']


class Babies(Feature):
    def create_features(self):
        self.train['babies'] = train['babies']
        self.test['babies'] = test['babies']


class Family_size(Feature):
    def create_features(self):
        self.train['family_size'] = train['adults'] + \
            train['children']+train['babies']
        self.test['family_size'] = test['adults'] + \
            test['children']+test['babies']



class Meal(Feature):
    def create_features(self):
        Meals = {"BB": 0, "HB": 1, "SC": 2, "FB": 3}

        self.train['meal'] = train['meal'].map(Meals)
        self.test['meal'] = test['meal'].map(Meals)


class Country(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'country'
        le.fit(pd.concat([train[col], test[col]]))
        self.train['country'] = le.transform(train[col])
        self.test['country'] = le.transform(test[col])


class Market_segment(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'market_segment'
        le.fit(pd.concat([train[col], test[col]]))
        self.train[col] = le.transform(train[col])
        self.test[col] = le.transform(test[col])


class Distribution_channel(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'distribution_channel'
        le.fit(pd.concat([train[col], test[col]]))
        self.train['country'] = le.transform(train[col])
        self.test['country'] = le.transform(test[col])


class Is_repeated_guest(Feature):
    def create_features(self):
        self.train['is_repeated'] = train['is_repeated_guest']
        self.test['is_repeated'] = test['is_repeated_guest']


class Previous_cancellations(Feature):
    def create_features(self):
        self.train['pre_cancell'] = train['previous_cancellations']
        self.test['pre_cancell'] = test['previous_cancellations']


class Previous_bookings_not_canceled(Feature):
    def create_features(self):
        self.train['pre_book_not_cancell'] = train['previous_bookings_not_canceled']
        self.test['pre_book_not_cancell'] = test['previous_bookings_not_canceled']


class Reserved_room_type(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'reserved_room_type'
        le.fit(pd.concat([train[col], test[col]]))
        self.train[col] = le.transform(train[col])
        self.test[col] = le.transform(test[col])


class Assigned_room_type(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'assigned_room_type'
        le.fit(pd.concat([train[col], test[col]]))
        self.train[col] = le.transform(train[col])
        self.test[col] = le.transform(test[col])


class Booking_changes(Feature):
    def create_features(self):
        self.train['booking_changes'] = train['booking_changes']
        self.test['booking_changes'] = test['booking_changes']


class Agent(Feature):
    def create_features(self):
        self.train['agent'] = train['agent']
        self.test['agent'] = test['agent']


class Company(Feature):
    def create_features(self):
        self.train['company'] = train['company']
        self.test['company'] = test['company']


class Day_in_waiting_list(Feature):
    def create_features(self):
        self.train['days_in_waiting_list'] = train['days_in_waiting_list']
        self.test['days_in_waiting_list'] = test['days_in_waiting_list']


class Customer_type(Feature):
    def create_features(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        col = 'customer_type'
        le.fit(pd.concat([train[col], test[col]]))
        self.train[col] = le.transform(train[col])
        self.test[col] = le.transform(test[col])


class Adr(Feature):
    def create_features(self):
        self.train['adr'] = train['adr']
        self.test['adr'] = test['adr']


class Required_car_parking_spaces(Feature):
    def create_features(self):
        self.train['required_car_parking_spaces'] = train['required_car_parking_spaces']
        self.test['required_car_parking_spaces'] = test['required_car_parking_spaces']


class Total_of_special_requests(Feature):
    def create_features(self):
        self.train['total_of_special_requests'] = train['total_of_special_requests']
        self.test['total_of_special_requests'] = test['total_of_special_requests']


if __name__ == '__main__':
    args = get_arguments()

    # train = pd.read_feather('./data/input/train.feather')
    # test = pd.read_feather('./data/input/test.feather')

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)
