#install.packages("e1071")
#install.packages("caret")
#install.packages("rgl")

library(caret)
library(e1071)

setwd('D:\\Users\\aedaniel\\Documents\\MCPI\\Minería de Datos\\SVM')

data <- read.csv('data_banknote_authentication.csv')

data
str(data)
summary(data)

is.na(data)
sum(is.na(data))

data$class <- as.factor(data$class)
data$class
table(data$class) 
barchart(table(data$class), horizontal = F)

hist(data$variance)
hist(data$entropy)
hist(data$skewness)
hist(data$curtosis)

boxplot(data$variance)
boxplot(data$entropy)
boxplot(data$skewness)
boxplot(data$curtosis)

set.seed(123456)
ids_train <- createDataPartition(data$class, p=0.7, list = F)
nrow(ids_train)
ids_train

modelo <- svm(class ~ ., data = data[ids_train,], cost=5)


# Utilizar cuando nuestro dataset no está equiñibrado en cuanto al número de muestras por clase
#modelo <- svm(class ~ ., data = data[ids_train,], class.weights = c('0'=0.4,'1'=0.6))
modelo
str(modelo)

table(data[ids_train,"class"], modelo$fitted, dnn=c('Actual','Predicho'))

prediccion <-predict(modelo, data[-ids_train,1:4])
table(data[-ids_train,"class"], prediccion, dnn=c('Actual','Predicho'))

plot(modelo, data=data[ids_train,], skewness ~ variance, col=c("magenta", "cyan"))
plot(modelo, data=data[-ids_train,], entropy ~ curtosis, col=c("magenta", "cyan"))


library(rgl)

colores <- c('red','green', 'blue')

plot3d(
  x=data[ids_train,'skewness'],
  y=data[ids_train,'variance'],
  z=data[ids_train,'curtosis'],
  col = colores,
  type = 's',
  radius = .1
)

svm_tune <- tune("svm", class ~ ., data=data, kernel='linear',
                 ranges = list(cost=c(0.001, 0.01,0.1, 1, 5, 10, 20, 50, 100, 150, 200)))

plot(svm_tune)
summary(svm_tune)                 

str(svm_tune)

best_model <- svm_tune$best.model

best_model
