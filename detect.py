import json
from dateutil import parser
import datetime
import pytz
import math
import sys
import argparse

threshold = 0.7
anom_free_days = 1
mode = 1
queue = 10 # Default value for unlimited queue
update = False
debug_out = False

argparser = argparse.ArgumentParser(description='Detection.')
argparser.add_argument('-t','--thresh', help='Similarity threshold.', required=False)
argparser.add_argument('-r','--retrain', help='Retrain length (days).', required=False)
argparser.add_argument('-m','--mode', help='1 .. default, 2 .. idf, 3 .. norm', required=False)
argparser.add_argument('-q','--queue', help='Queue size (-1 for unlimited).', required=False)
argparser.add_argument('-u','--update', help='Update model also during detection.', required=False, action='store_true')
argparser.add_argument('-d','--debug', help='Output debug information.', required=False, action='store_true')

args = vars(argparser.parse_args())
if args["thresh"] is not None:
    threshold = float(args["thresh"])
if args["retrain"] is not None:
    anom_free_days = int(args["retrain"])
if args["mode"] is not None:
    mode = int(args["mode"]) # 1 .. normal, 2 .. idf, 3 .. norm
if args["queue"] is not None:
    queue = int(args["queue"])
update = args["update"]
debug_out = args["debug"]

# Read in ground truth (switched uid and corresponding timestamps)
anomalous_users = {}
with open('labels.txt') as f:
    print('Ground truth: ')
    for line in f:
        parts = line.split(',')
        ts = datetime.datetime.fromtimestamp(int(float(parts[1])), tz=pytz.UTC)
        anomalous_users[parts[0]] = datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC) # Omit time info since we count detected days
        print(' * ' + str(parts[0]) + ' switched at ' + str(ts))

total_days = {}
wait_days = {}
cnt = 0
detected_dist = {}
last_anom_dist = {}
freq_day = {}
last_active_day = {}
dists = {}
debug = {}
idf = {}
only_anomalous_users = False # Skip normal users that are not in the ground truth (mainly for debugging/testing)
total_lines = 50522931
with open('clue_anomaly.json') as f:
    for line in f:
        cnt += 1
        if int(cnt % (total_lines / 20)) == 0:
            print(str(int(cnt*100/total_lines)) + '%', end=' ', flush=True)
        j = json.loads(line)
        uid = j['uid']
        if only_anomalous_users and uid not in anomalous_users:
            # Only consider anomalous users and skip normal users for analysis
            continue
        action = j['type']
        # Count by how many users each event type is used for idf-weighting if mode is set to idf
        if action not in idf:
            idf[action] = set([uid])
        else:
            idf[action].add(uid)
        ts = parser.isoparse(j['time'])
        currentday = datetime.datetime(ts.year, ts.month, ts.day, 0, 0, 0, 0, pytz.UTC) # Omit hour, minute, and second
        # Store all days where each user is active
        if uid not in total_days:
            total_days[uid] = set([currentday])
        else:
            total_days[uid].add(currentday)
        if uid not in last_active_day:
            # First appearance of uid - initialize user information
            freq_day[uid] = {}
            freq_day[uid][action] = 1
            wait_days[uid] = anom_free_days
            debug[uid] = []
            last_active_day[uid] = currentday
            dists[uid] = []
        else:
            if last_active_day[uid] != currentday:
                # Start of a new day - check count vector of previous day
                min_dist = None
                min_known = None
                min_limit = None
                duplicate_check = []
                for known in dists[uid]:
                    # Check all count vectors present in model
                    if known in duplicate_check:
                        # Identical count vector already checked, no need to check again
                        continue
                    duplicate_check.append(known)
                    # Initialize distance measures
                    manh = 0
                    limit = 0
                    for action_element in set(list(known.keys()) + list(freq_day[uid].keys())):
                        # Go through all event types that are either in the model count vector or checked count vector
                        idf_fact = 1
                        if mode == 2:
                            # Weigh event types lower when many users use them
                            idf_fact = math.log10((1 + len(last_active_day)) / len(idf[action_element]))
                        norm_sum_known = 1
                        norm_sum_freq = 1
                        if mode == 3:
                            # Normalize event count vectors so that only relative frequencies matter
                            norm_sum_known = sum(known.values())
                            norm_sum_freq = sum(freq_day[uid].values())
                        # Increase distance measures depending on how often each event type occurs in model and checked count vector
                        if action_element not in known:
                            # Event type only in checked count vector - increase distance by count value
                            manh += freq_day[uid][action_element] * idf_fact / norm_sum_freq
                            limit += freq_day[uid][action_element] * idf_fact / norm_sum_freq
                        elif action_element not in freq_day[uid]:
                            # Event type only in model count vector - increase distance by count value
                            manh += known[action_element] * idf_fact / norm_sum_known
                            limit += known[action_element] * idf_fact / norm_sum_known
                        else:
                            # Event type in both model and checked count vector - increase distance by the absolute difference (0 for perfect match)
                            manh += abs(freq_day[uid][action_element] * idf_fact / norm_sum_freq - known[action_element] * idf_fact / norm_sum_known)
                            # Increase limit by maximum of both count values as it is the upper limit for the distance
                            limit += max(freq_day[uid][action_element] * idf_fact / norm_sum_freq, known[action_element] * idf_fact / norm_sum_known)
                    if min_dist is None:
                        # Initialize minimum distance
                        min_dist = manh / limit # anomaly score (how close is distance to upper limit)
                        min_known = known # most similar vector from model
                        #limit = sum(list(map(add, norm, known))) # max(sum(norm), sum(known_norm)) / 2
                        min_limit = limit
                    else:
                        # Check for smaller distance
                        if manh / limit < min_dist:
                            min_dist = manh / limit
                            min_known = known
                            min_limit = limit
                    if min_dist == 0:
                        # Perfect match, no need to go over all other elements
                        break
                # Get date of previous (and currently analyzed) day
                last_active_day_date = datetime.datetime(last_active_day[uid].year, last_active_day[uid].month, last_active_day[uid].day, 0, 0, 0, 0, pytz.UTC)
                # Each user on each day where they are active is subject of detection
                sample = (uid, last_active_day_date)
                if sample not in detected_dist:
                    detected_dist[sample] = []
                if wait_days[uid] <= 0:
                    # Detection is currently ongoing as sufficient days have passed since last anomaly
                    detected_dist[sample].append("detection")
                else:
                    # Re-training is currently ongoing
                    detected_dist[sample].append("training")
                if min_dist is not None and min_dist > threshold:
                    # Model is empty or count vector is considered anomalous
                    detected_dist[sample].append("anomalous")
                    # Restart re-training
                    wait_days[uid] = anom_free_days
                else:
                    # Count vector is considered normal
                    detected_dist[sample].append("normal")
                    # Reduce days left for re-training
                    wait_days[uid] -= 1
                if update is True or wait_days[uid] >= 0:
                    # Add new count vector to model when model is updated also for normal data or currently re-training
                    dists[uid].append(freq_day[uid])
                if queue != -1 and len(dists[uid]) >= queue:
                    # Remove oldest count vector from queue in model
                    dists[uid] = dists[uid][1:]
                if debug_out is True:
                    debug[uid].append((datetime.datetime(last_active_day[uid].year, last_active_day[uid].month, last_active_day[uid].day, 0, 0, 0, 0, pytz.UTC), min_dist, min_limit, detected_dist[sample], min_known, freq_day[uid]))
                last_active_day[uid] = currentday
                freq_day[uid] = {}
            if action in freq_day[uid]:
                freq_day[uid][action] += 1
            else:
                freq_day[uid][action] = 1

if debug_out is True:
    # Print anomaly scores for each user and each day
    with open('debug.txt', 'w+') as out:
        for uid, lst in debug.items():
            if len(lst) > 15: # Skip users with few active days
                out.write('\n' + uid + '\n')
                for elem in lst:
                    string = ""
                    if uid in anomalous_users and anomalous_users[uid].timestamp() <= elem[0].timestamp():
                        string += " Changed user!"
                    if elem[1] is not None and elem[2] is not None and elem[1] > elem[2]:
                        string += ' Detected!'
                    if elem[1] is not None and elem[2] is not None:
                        out.write(str(elem[0]) + ': ' + str(elem[1] * elem[2]) + '/' + str(elem[2]) + ' #' + str(round(elem[1], 2)) + string + ' ' + str(elem[3]) + '\n')
                    if uid in anomalous_users and abs(anomalous_users[uid].timestamp() - elem[0].timestamp()) < 60*60*24*60 and elem[4] is not None and elem[5] is not None:
                        # Also print count vectors for days close to switching anomalous users
                        out.write(str(dict(sorted(elem[4].items()))) + '\n')
                        out.write(str(dict(sorted(elem[5].items()))) + '\n')

def get_eval_results(d):
    # Initialize metrics
    tp = 0
    tp_adjusted = 0
    fp = 0
    tn = 0
    fn = 0
    tp_user = {}
    tp_adjusted_user = {}
    fn_user = {}
    for uid in anomalous_users:
        # Count detections separately for each anomalous user
        tp_user[uid] = 0
        tp_adjusted_user[uid] = 0
        fn_user[uid] = 0
    sum_training = 0
    sum_detection = 0
    for tup, detected in d.items():
        if "training" in detected:
            # Count all days for all users spent for (re-)training
            sum_training += 1
        elif "detection" in detected:
            # Count all days for all users spent for detecting
            sum_detection += 1
        uid = tup[0]
        day = tup[1]
        # Check if detected user is in ground truth, the timestamp is after the switching point, and that only first day after switch is considered
        if uid in anomalous_users and day.timestamp() == anomalous_users[uid].timestamp():
            if detected == ["detection", "anomalous"]:
                # Normal correct detection during detecting phase
                tp += 1
                tp_user[uid] += 1
                tp_adjusted += 1
                tp_adjusted_user[uid] += 1
            elif detected == ["detection", "normal"] or detected == ["training", "normal"]:
                # Missed anomalous user (classified as normal) either during training or detection phase
                fn += 1
                fn_user[uid] += 1
            elif detected == ["training", "anomalous"]:
                # Correct detection during training phase counted by adjusted score
                tp_adjusted += 1
                tp_adjusted_user[uid] += 1
        else:
            # Note that instances are omitted in the training phase
            if detected == ["detection", "anomalous"]:
                # Incorrect detection of normal behavior during detection phase
                fp += 1
            elif detected == ["detection", "normal"]:
                # Correctly non-detected instance during detection phase
                tn += 1
    # Print all metrics to console
    print('  Total = ' + str(tp + tn + fp + fn))
    print('  Train = ' + str(sum_training))
    print('  Detect = ' + str(sum_detection))
    print('  TP_adj = ' + str(tp_adjusted))
    print('  TP = ' + str(tp))
    for uid in anomalous_users:
        print(' - ' + str(uid) + ': TP_adj = ' + str(tp_adjusted_user[uid]) + ', FN = ' + str(fn_user[uid]))
    print('  FP = ' + str(fp))
    print('  TN = ' + str(tn))
    print('  FN = ' + str(fn))
    tpr_adjusted = "NaN"
    if tp_adjusted + fn > 0:
        tpr_adjusted = tp_adjusted / (tp_adjusted + fn)
    print('  TPR_adj = ' + str(tpr_adjusted))
    tpr = "NaN"
    if tp + fn > 0:
        tpr = tp / (tp + fn)
    print('  TPR = Rec = ' + str(tpr))
    fpr = "NaN"
    if fp + tn > 0:
        fpr = fp / (fp + tn)
    print('  FPR = ' + str(fpr))
    tnr = "NaN"
    if tn + fp > 0:
        tnr = tn / (tn + fp)
    print('  TNR = ' + str(tnr))
    prec = "NaN"
    if tp_adjusted + fp > 0:
        prec = tp_adjusted / (tp_adjusted + fp)
    print('  Prec = ' + str(prec))
    fone = "NaN"
    if tp_adjusted + 0.5 * (fp + fn) > 0:
        fone = tp_adjusted / (tp_adjusted + 0.5 * (fp + fn))
    print('  F1 = ' + str(fone))
    acc = "NaN"
    if tp_adjusted + tn + fp + fn > 0:
        acc = (tp_adjusted + tn) / (tp_adjusted + tn + fp + fn)
    print('  ACC = ' + str(acc))
    print('  R = ' + str(sum_training / (sum_training + sum_detection)))
    print('thresh,retrain,mode,queue,update,total,train,detect,tp_adj,tp,fp,tn,fn,tpr_adj,tpr,fpr,tnr,p,f1,acc')
    print(str(threshold) + ',' + str(anom_free_days) + ',' + str(mode) + ',' + str(queue) + ',' + str(update) + ',' + str(tp + tn + fp + fn) + ',' + str(sum_training) + ',' + str(sum_detection) + ',' + str(tp_adjusted) + ',' + str(tp) + ',' + str(fp) + ',' + str(tn) + ',' + str(fn) + ',' + str(tpr_adjusted) + ',' + str(tpr) + ',' + str(fpr) + ',' + str(tnr) + ',' + str(prec) + ',' + str(fone) + ',' + str(acc))
    print('')

sum_total_days = 0
for uid, total_day_count in total_days.items():
    sum_total_days += len(total_day_count)
print('\n ' + str(len(total_days)) + ' users with ' + str(sum_total_days) + ' days considered, including days spent on training and incomplete days.')

print('Results with threshold = ' + str(threshold) + ':')
get_eval_results(detected_dist)

