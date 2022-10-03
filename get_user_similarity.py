import json
from dateutil import parser
import datetime
import pytz
import argparse

argparser = argparse.ArgumentParser(description='Similarity computation.')
argparser.add_argument('-d','--min_days', help='Minimum amount of days user must be active.', required=False)
argparser.add_argument('-c','--w_count', help='Weight of average user event count per day for similarity computation.', required=False)
argparser.add_argument('-e','--w_events', help='Weight of shared event types for similarity computation.', required=False)

min_days = 25
w_count = 0.3
w_events = 0.7

args = vars(argparser.parse_args())
if args["min_days"] is not None:
    min_days = int(args["min_days"])
if args["w_count"] is not None:
    w_count = float(args["w_count"])
if args["w_events"] is not None:
    w_events = float(args["w_events"])

# Normalize weights in case that they do not sum to 1
w_sum = w_count + w_events
w_count /= w_sum
w_events /= w_sum

users = {}
cnt = 0
total_lines = 50522931

# Go through all lines and retrieve information on executed events for each user
with open('clue.json') as f:
    for line in f:
        cnt += 1
        if cnt % (total_lines / 20) == 0:
            print(str(int(cnt*100/total_lines)) + '%', end=' ', flush=True)
        j = json.loads(line)
        uid = j['uid']
        action = j['type']
        ts = parser.isoparse(j['time'])
        if uid not in users:
            users[uid] = {'actions': set([action]), 'cnt': 1, 'first': ts.timestamp(), 'last': ts.timestamp(), 'day_list': set([datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC).timestamp()]), 'days': 1}
        else:
            users[uid]['actions'].add(action)
            users[uid]['cnt'] += 1
            users[uid]['last'] = ts.timestamp()
            users[uid]['day_list'].add(datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC).timestamp())
            users[uid]['days'] = len(users[uid]['day_list'])
print('')

# Compute similarities between pairs of users and store them in a similarity matrix
similarity_matrix = {}
user_list = []
for user in users:
    users[user]['similarities'] = [] # use list instead of dict to reduce resulting file size
    user_list.append(user) # this list is necessary to map user names to list of similarities
    for user_inner in users:
        # Compute similarity based on average number of generated events
        cnt_sim = min(users[user]['cnt'] / users[user]['days'], users[user_inner]['cnt'] / users[user_inner]['days']) / max(users[user]['cnt'] / users[user]['days'], users[user_inner]['cnt'] / users[user_inner]['days'])
        # Compute similarity based on common event types
        action_sim = len(users[user]['actions'].intersection(users[user_inner]['actions'])) / len(users[user]['actions'].union(users[user_inner]['actions']))
        # Compute weighted similarity score
        sim_score = w_count * cnt_sim + w_events * action_sim
        users[user]['similarities'].append(round(sim_score, 3)) # round to decrease resulting file size
        if users[user]['days'] > min_days and users[user_inner]['days'] > min_days:
            if user not in similarity_matrix:
                similarity_matrix[user] = {}
            similarity_matrix[user][user_inner] = sim_score

# Output the similarity matrix
with open('sim.txt', 'w+') as out:
    for user in similarity_matrix:
        string = ""
        for user_inner in similarity_matrix[user]:
            string += str(similarity_matrix[user][user_inner]) + ','
        out.write(string[:-1] + '\n')

# Convert sets to lists to enable serialization
for user in users:
    users[user]['actions'] = list(users[user]['actions'])
    users[user]['day_list'] = list(users[user]['day_list'])

# Output detailed information for further processing
with open('user_info.txt', 'w+') as out:
    out.write(json.dumps({'user_list': user_list, 'user_info': users}))    

