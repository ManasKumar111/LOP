import cv2
import numpy as np
import pandas as pd

# Define the region of interest (ROI) as a tuple of (x, y, w, h) values
roi = (100, 100, 200, 200)

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([20, 100, 70])
upper = np.array([30, 255, 255])  # (These ranges will detect Yellow)

# Capturing webcam footage
webcam_video = cv2.VideoCapture(0)

# Create an empty list to store the cropped images
cropped_images = []

while True:
    success, video = webcam_video.read()  # Reading webcam footage

    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)  # Converting BGR image to HSV format

    # Crop the image to the ROI
    x, y, w, h = roi
    img = img[y:y + h, x:x + w]

    mask = cv2.inRange(img, lower, upper)  # Masking the image to find our color

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)  # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        rec = 0;
        for mask_contour in mask_contours:

            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3)  # drawing rectangle
                rec = rec + 1;
                if (rec >= 4):
                    # Save the cropped image to the list
                    cropped_image = img[y:y + h, x:x + w]
                    cropped_images.append(cropped_image)

    cv2.imshow("window", video)  # Displaying webcam image

    # If we have collected enough cropped images, stop capturing video
    if len(cropped_images) >= 100:
        break

    cv2.waitKey(1)

webcam_video.release()
cv2.destroyAllWindows()

# Create a DataFrame to store the concentration and temperature values
df = pd.DataFrame(columns=["Concentration", "Temperature"])

# Loop over the cropped images and prompt the user to enter the concentration and temperature values
for cropped_image in cropped_images:
    # Display the cropped image and prompt the user to enter the concentration and temperature values
    cv2.imshow("cropped image", cropped_image)
    concentration = input("Enter the concentration value: ")
    temperature = input("Enter the temperature value: ")

    # Add the concentration and temperature values to the DataFrame
    df = df.append({"Concentration": concentration, "Temperature": temperature}, ignore_index=True)

    cv2.waitKey(1)

cv2.destroyAllWindows()

# Train the ML model using the concentration, temperature, and cropped image data
# (insert your own code for this step here)
# Train the ML model using the concentration, temperature, and cropped image data
# (insert your own code for this step here)

# Load the jeff_reaction_data.csv file into a Pandas DataFrame
data = pd.read_csv("jeff_reaction_data.csv")

# Filter the DataFrame to only include data from the specific region of interest
data = data.loc[(data["X"] > roi[0]) & (data["X"] < roi[0] + roi[2]) & (data["Y"] > roi[1]) & (data["Y"] < roi[1] + roi[3])]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)

# Extract the cropped images from the training and testing data
train_images = []
test_images = []
for index, row in train_data.iterrows():
    img = cv2.imread(row["Image_Path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    train_images.append(img)
for index, row in test_data.iterrows():
    img = cv2.imread(row["Image_Path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    test_images.append(img)

# Extract the concentration and temperature values from the training and testing data
train_concentrations = train_data["Concentration"]
test_concentrations = test_data["Concentration"]
train_temperatures = train_data["Temperature"]
test_temperatures = test_data["Temperature"]

# Reshape the image data into the correct format for training the model
train_images = np.array(train_images)
test_images = np.array(test_images)
train_images = train_images.reshape(train_images.shape[0], roi[2]*roi[3])
test_images = test_images.reshape(test_images.shape[0], roi[2]*roi[3])

# Normalize the image data
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Train the ML model using the concentration, temperature, and image data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_images, train_concentrations)

# Test the ML model using the testing data
from sklearn.metrics import mean_squared_error, r2_score
test_predictions = model.predict(test_images)
test_mse = mean_squared_error(test_concentrations, test_predictions)
test_r2 = r2_score(test_concentrations, test_predictions)
print("Test MSE: ", test_mse)
print("Test R2 Score: ", test_r2)

# Visualize the predicted concentrations as a function of temperature
import matplotlib.pyplot as plt

# Generate a range of temperature values to use for plotting the predicted concentrations
temperature_range = np.linspace(min(test_temperatures), max(test_temperatures), 100)

# Generate predicted concentrations for each temperature value in the range
concentration_predictions = []
for temp in temperature_range:
    img = np.zeros((roi[3], roi[2]), dtype="uint8")
    img.fill(int(temp))
    img = img.reshape(1, roi[2]*roi[3])
    concentration_prediction = model.predict(img)
    concentration_predictions.append(concentration_prediction[0])

# Plot the predicted concentrations as a function of temperature
plt.scatter(test_temperatures, test_concentrations, color="blue", label="Actual Concentrations")
plt.plot(temperature_range, concentration_predictions, color="red", label="Predicted Concentrations")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Concentration")
plt.title("Predicted Concentrations vs. Temperature")
plt.legend()
plt.show()
