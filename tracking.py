import math
import time

FRAMES_NOT_SEEN_BUFFER = 5

class Vehicle:
    def __init__(self, top, bottom, left, right, id, exit_time=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.id = id
        self.entry_time = time.time()
        self.exit_time = exit_time
        self.buffer = FRAMES_NOT_SEEN_BUFFER

    def calulate_speed(self, distace):
        if self.entry_time == self.exit_time:
            return 0
        velocity = distace/(self.exit_time - self.entry_time)
        return velocity

def update_or_deregister(objects, vehicles, distance):
    indexes_to_be_deleted = []
    for i in range(len(vehicles)):
        best_match, best_match_distance = None, 1e9
        bxmin = vehicles[i].left
        bymin = vehicles[i].top
        bxmax = vehicles[i].right
        bymax = vehicles[i].bottom
        bxmid = (bxmin + bxmax) / 2
        bymid = (bymin + bymax) / 2
        for j in range(len(objects)):
            top = objects[j][0]
            bottom = objects[j][1]
            ymid = int(round((top + bottom) / 2))
            left = objects[j][2]
            right = objects[j][3]
            xmid = int(round((left + right) / 2))
            box_range = ((right - left) + (bottom - top)) / 2 + 10

            distance = math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2)
            if  distance < box_range and distance < best_match_distance:
                best_match = objects[j]
                best_match_distance = distance

        if best_match == None: # Mark for delete
            indexes_to_be_deleted.append(i)
        else: #Update
            vehicles[i].top = best_match[0]
            vehicles[i].bottom = best_match[1]
            vehicles[i].left = best_match[2]
            vehicles[i].right = best_match[3]
            vehicles[i].buffer = FRAMES_NOT_SEEN_BUFFER

    vehicle_velocity_sum, deleted_counts = 0, 0

    for index in sorted(indexes_to_be_deleted, reverse=True):
        if vehicles[index].buffer == 0:
            vehicles[index].exit_time = time.time()
            vehicle_velocity_sum += vehicles[index].calulate_speed(distance)
            deleted_counts += 1
            del vehicles[index]
        else:
            vehicles[index].buffer -= 1

    return vehicle_velocity_sum, deleted_counts

def not_tracked(objects, vehicles, v_count): # Will return new objects
    if len(objects) == 0:
        return []  # No new classified objects to search for

    new_vehicles = []
    for obj in objects:
        top = obj[0]
        bottom = obj[1]
        ymid = int(round((top+bottom)/2))
        left = obj[2]
        right = obj[3]
        xmid = int(round((left+right)/2))
        box_range = ((right - left) + (bottom - top)) / 2 + 10
        for vehicle in vehicles + new_vehicles:
            bxmin = vehicle.left
            bymin = vehicle.top
            bxmax = vehicle.right
            bymax = vehicle.bottom
            bxmid = (bxmin + bxmax) / 2
            bymid = (bymin + bymax) / 2
            if math.sqrt((xmid - bxmid)**2 + (ymid - bymid)**2) < box_range:
                # found existing, so break (do not add to new_objects)
                break
        else:
            new_vehicles.append(Vehicle(obj[0], obj[1], obj[2], obj[3], v_count + 1))
            v_count += 1

    return new_vehicles