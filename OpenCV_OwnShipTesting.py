import gymnasium as gym
import ale_py
import cv2


gym.register_envs(ale_py)
#env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
#obs, info = env.reset()
#episode_over = False
#count = 0
#env.close()

image = cv2.imread("Images/image96.png")
print(image.shape)
image2 = image[175:193,0:160]
edges = cv2.Canny(image2, 100, 200)
contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("Canny", edges)
cv2.waitKey(0)
count = 0
for contour in contours:
    #print(contour)
    count = count + 1
    moment = cv2.moments(contour)
    if moment["m00"] != 0:
        x_val = int(moment["m10"] / moment["m00"])
        y_val = int(moment["m01"] / moment["m00"])

print(count)
#print(moments['m00'])
x_val = (moment['m10']/moment['m00'])
print(x_val)
cv2.circle(image,(int(x_val),570),5,(0,0,225),-1)
#cv2.imshow("Canny", contours)
#cv2.waitKey(0)
cv2.imshow("Image", image)
cv2.waitKey(0)

print(type(image))