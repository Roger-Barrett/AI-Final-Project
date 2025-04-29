import math
import random
import gymnasium as gym
import ale_py
import cv2
import numpy as np
from PIL import Image


class ApproximateQAgent(object):
    def __init__(self):
        self.q_table = {}
        self.epsilon = 0.2
        self.alpha = 0.5
        self.discount = 0.5
        self.weights = [1,1,1,1]

    def getWeights(self):
        return self.weights
    def getQValue(self,state,action):
        features = self.feature_extraction(state)
        #print(features)
        sum = 0
        for feature in range(len(features)):
            sum = sum + self.weights[feature] * features[feature]
        return sum


    def computeValuefromQValue(self,state):
        actions = [0,1,2,3,4]
        best_action = None
        max_value = -1000
        for action in actions:
            value = self.getQValue(state,action)
            if value > max_value:
                max_value = value
                best_action = action
        return max_value

    def computeActionfromQValue(self,state):
        actions = [0,1,2,3,4,5]
        best_action = None
        max_value = -1000
        for action in actions:
            value = self.getQValue(state,action)
            if action == 0:
                value = value - 5
            if action == 1:
                value = value + (self.weights[1]/1)*5/300
            if action == 2:
                value = value + self.weights[0]*13/80 -5
            if action == 3:
                value = value + self.weights[0]*13/80 -5
            if action == 4:
                value = value + (self.weights[1]/1)*5/300+ self.weights[0]*13/80
            if action == 5:
                value = value + (self.weights[1]/1)*5/300 + self.weights[0]*13/80
            if value > max_value:
                max_value = value
                best_action = action
        return best_action

    def getAction(self,state):
        actions = [0,1,2,3,4]
        action = None
        if random.random() > self.epsilon:
            action = random.choice(actions)
        else:
            action = self.computeActionfromQValue(state)
        if action == None:
            action = random.choice(actions)
        return action

    def update(self,previous_state,action,state,reward):
        features = ApproximateQAgent.feature_extraction(self,previous_state)
        current_q_value = self.getQValue(previous_state,action)
        difference = (reward + self.discount * self.computeValuefromQValue(state)) - current_q_value
        for feature in range(len(features)):
            update = self.alpha * difference * features[feature]
            self.weights[feature] += self.weights[feature] * update
        return

    def feature_extraction(self,obs):
        own_ship_image = obs[175:193,0:160]
        own_ship_edges = cv2.Canny(own_ship_image,100,200)
        own_ship_contours, _ = cv2.findContours(own_ship_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        own_ship_x_val = 80
        for contour in own_ship_contours:
            moment = cv2.moments(contour)
            if moment['m00'] != 0:
                own_ship_x_val = moment['m10']/moment['m00']

        invaders_image = obs[25:150,0:160]
        invaders_edges = cv2.Canny(invaders_image,100,200)
        invaders_contours, _ = cv2.findContours(invaders_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        closest_invader_x = 1000
        closest_invader_y = 1000
        closest_invader_distance = 10000
        invader_count = 0
        for contour in invaders_contours:
            invader_count += 1
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                x_val = moments['m10']/moments['m00']
                y_val = moments['m00']/moments['m00']
                current_distance = math.sqrt((own_ship_x_val-x_val)**2 + (y_val-180+25)**2)
                if current_distance < closest_invader_distance:
                    closest_invader_x = x_val
                    closest_invader_y = y_val
                    closest_invader_distance = current_distance
        barrier_image = obs[150:177,0:160]
        barrier_edges = cv2.Canny(barrier_image,100,200)
        barrier_contours,_ = cv2.findContours(barrier_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        barrier_area = 0
        for contour in barrier_contours:
            barrier_area = barrier_area + cv2.contourArea(contour)
        #print("x" , own_ship_x_val)
        #print("close", closest_invader_x)
        #print("invader_count" ,invader_count)
        #print("Area" ,barrier_area)
        return [abs(own_ship_x_val-80)/160, (invader_count/200), (closest_invader_distance/230), (barrier_area/600)]



