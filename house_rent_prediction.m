clc; clear; close all;

% Data
Size = [100, 130, 80, 150, 120]';
Rooms = [3, 4, 2, 3, 3]';
Ages = [10, 15, 5, 20, 8]';
Rent = [20000, 25000, 16000, 28000, 23000]';

% Input
X = [Size, Rooms, Ages];
Y = Rent;

% Linear Regression Model
linearModel = fitlm(X, Y);
disp(linearModel);

% User-defined affordability threshold
budget = input('Enter your budget for rent: ');
Y_class = Y <= budget;  % 1 if affordable, 0 if not

% Logistic Regression Model
logisticModel = fitglm(X, Y_class, 'Distribution', 'binomial', 'LikelihoodPenalty', 'jeffreys-prior');

% New data input
newSize = input('Enter house size (m²): ');
newRooms = input('Enter number of rooms: ');
newAges = input('Enter house age (years): ');

newData = [newSize, newRooms, newAges];

% Rent prediction
predictedRent = predict(linearModel, newData);
fprintf('Predicted Rent: %.2f\n', predictedRent);

% Affordability classification
predictedClass = predictedRent <= budget;
if predictedClass == 1
    fprintf('Predicted Affordability: AFFORDABLE\n');
else
    fprintf('Predicted Affordability: NOT AFFORDABLE\n');
end

% Visualization
figure;

subplot(2,1,1);
scatter(Size, Rent, 80, Ages, 'filled');
hold on;
scatter(newSize, predictedRent, 120, 'rx', 'LineWidth', 3);

SizeRange = linspace(min(Size), max(Size), 100)';
RoomsMean = mean(Rooms) * ones(size(SizeRange));
AgesMean = mean(Ages) * ones(size(SizeRange));
RentPredicted = predict(linearModel, [SizeRange, RoomsMean, AgesMean]);
plot(SizeRange, RentPredicted, 'b-', 'LineWidth', 2);

legend('Data (Colored by Age)', 'New Prediction', 'Regression Line', 'Location', 'best');
title('Linear Regression Model');
xlabel('Size (m²)');
ylabel('Predicted Rent');
colorbar;
grid on;
hold off;

subplot(2,1,2);
hold on;
scatter(Y, Y_class, 80, 'bo', 'filled');
scatter(predictedRent, predictedClass, 120, 'rx', 'LineWidth', 3);

legend('Existing Data', 'New Prediction', 'Location', 'best');
xlabel('Actual Rent');
ylabel('Affordability Probability (0 to 1)');
title('Logistic Regression');
grid on;
hold off;
