############################################################################################
# First part of R script is dedicated to import data and adjust variables to make ir processable
############################################################################################
###########################################################################################
#Carrega os pacotes a serem utilizados
#PASCKAGES INSTALLATION CODES
install.packages("caret")
install.packages("AppliedPredictiveModeling")
install.packages("mlbench")
install.packages("QSARdata")
install.packages("ipred")
install.packages("parallelRandomForest")
install.packages("e1071")
install.packages("pROC")
install.packages("gbm")
install.packages("doParallel")
install.packages("reshape")
install.packages("foreign")
install.packages("readstata13")
install.packages("data.table")

#Load packages needed for the analysis
lapply(c("caret","AppliedPredictiveModeling","mlbench","QSARdata","ipred","parallelRandomForest","e1071","pROC","gbm","doParallel", "reshape", "foreign",
         "readstata13", "data.table"),library, character.only=T)

############################################################################################
# Import data from STATA
############################################################################################
setwd("E:/Duke university/Artigos Joao/Modelo preditivo Ben/database nao testado")

#load all datasets simultaneously (take care to not overload memory limits)

#data.table version
#files <- list.files(path = "E:/Duke university/Artigos Joao/Modelo preditivo Ben/database nao testado/",pattern = ".dta")
#nome_factor <- nchar(files)-4
#factors <- substr(files, start=1, stop= nome_factor)
#rm(nome_factor)

# select variables to be used in this model
col_selected <- c('male_01',
                  'ethnicity_01',
                  'race_01',
                  'agec_01',
                  'prem_01',
                  'height_to_age_01',
                  'weight_to_age_01',
                  'height_to_weight_01',
                  'crf_01',
                  'prvpcs_01',
                  'diabetes_0',
                  'wtloss_01',
                  'nutr_support_01',
                  'asthma_01',
                  'struct_pulm_ab_01',
                  'oxygen_sup_01',
                  'ventilat_01',
                  'tracheostomy_01',
                  'hxcld_01',
                  'cystic_fib_01',
                  'cpneumon_01',
                  'bleeddis_01',
                  'hemodisorder_01',
                  'transfus_01',
                  'immune_dis_01',
                  'steroid_01',
                  'cancer_01',
                  'chemo_01',
                  'radio_01',
                  'bone_marrow_trans_01',
                  'organ_trans_01',
                  'acq_abnormality_01',
                  'cerebral_palsy_01',
                  'tumorcns_01',
                  'coma_01',
                  'neuromuscdis_01',
                  'impcogstat_01',
                  'seizure_01',
                  'ivh_01',
                  'cva_01',
                  'esovar_01',
                  'lbp_disease_01',
                  'renafail_01',
                  'dialysis_01',
                  'wndinf_01',
                  'proper30_01',
                  'preop_sepsis_1',
                  'inotr_support_01',
                  'cpr_prior_surg_01',
                  'dnr_01',
                  'htooday',
                  'tothlos_01',
                  'workrvu',
                  'optime_01',
                  'asaclas_01',
                  'wndclas_01',
                  'transt_01',
                  'inpatient_01',
                  'casetype_01',
                  'adverse_any_01')


for (i in 1:length(files)) {
  temp <- read.dta13(paste("E:/Duke university/Artigos Joao/Modelo preditivo Ben/database nao testado/",files[i], sep = ""),convert.factors = TRUE, encoding = "UTF-8", add.rownames = TRUE, nonint.factors = TRUE)
  temp <- as.data.table(temp)
  temp <- subset(temp,,col_selected)
  temp$procedure <- as.factor(rep(factors[i],length(temp)))
  if (!exists("adverse_event")){
    adverse_event <- temp}
  else {adverse_event <- rbind(adverse_event, temp)}
}

rm(temp, col_selected,factors,files,i)

############################################################################################
# Begin of preparation for predictive modelling
############################################################################################
# I did the option to use step by step approach instead of use caret preprocessing to make easier identify where may be any problems
#create dummy variables
head(model.matrix(~., data = adverse_event))
dummies <- dummyVars(~., data = adverse_event)
head(predict(dummies, newdata = adverse_event))
adverse_event <-predict(dummies, newdata = adverse_event)
adverse_event <- data.table(adverse_event)
rm(dummies)
#missing check - (there is some variables with large number of missings)
sapply(adverse_event, function(x) sum(is.na(x)))
adverse_event <- na.omit(adverse_event)

# Create outcome
outcome_adverse_event <- adverse_event$adverse_any_01.yes


#create a dataset only with predictors
predictors_adverse_event <- subset(adverse_event, select = -c(adverse_any_01.yes,adverse_any_01.no))

#Near-Zero variance - analysis (there is 78 of 125 variables with near zero variance issue)
nzv <- nearZeroVar(predictors_adverse_event, saveMetrics= TRUE)
summary(nzv$nzv)
predictors_adverse_event[, rownames(nzv[nzv$nzv,])] <- list(NULL)
rm(nzv)

#### Create partitioning datasets
# Training set  will be used to performed all modeling
# Test set will be used to test model's performance - accuracy, model comparison
# set a random number to start the randomization
set.seed(107)

# arguments are: dataset = nome do dataset, proportion = 3/4
inTrain <- createDataPartition(outcome_adverse_event, p = 3/4, list = FALSE)
trainDescr <- predictors_adverse_event[inTrain,] # isolate descriptors for the train data 
testDescr <- predictors_adverse_event[-inTrain,] # isolate descriptors for the test data 

trainClass <- outcome_adverse_event[inTrain] # isolate outcome for the train data
testClass <- outcome_adverse_event[-inTrain] # isolate outcome for the test data

rm(inTrain)

############################################################################
# Paralell Processing
############################################################################

doParallel::registerDoParallel(4) #define number of threads that will be used during analysis process

############################################################################
# Pre-Processing
############################################################################
##### Multicolinearity assessment
col_antes <- ncol(trainDescr) #number of columns in the dataset

#create correlation matrix
descrCorr <- cor(trainDescr)

#identify highly correlated data - above 0.90
highCorr <- findCorrelation(descrCorr, .90, verbose= T)

# exclude highly correlated data from the dataset
#turnover
trainDescr<- subset(trainDescr, select = -highCorr)
testDescr <- subset(testDescr, select = -highCorr)
#show the mumber of removed columns
ncol(trainDescr)- col_antes

rm(descrCorr, col_antes, highCorr)
##############################################################################################
#### Pre-process data to find center, re-scaling, dimensionality issues
# The function has an argument, method, that can have possible values of 
#"center", "scale", "pca" and "spatialSign". The first two options provide 
#simple location and scale transformations of each predictor (and are the 
# default values of method).

preprocess<- preProcess(trainDescr) #define pre-processing methods for each variable (precisa fazer duas vezes para cada porção da intrain?)

#Apply pre-processing to dataset
trainDescr <- predict(preprocess, trainDescr)
testDescr <- predict(preprocess, testDescr)

#finding linear combos
#turnover
comboInfo <- findLinearCombos(trainDescr)
comboInfo

#remove os combos lineares dos datasets de teste
trainDescr<-trainDescr[, -comboInfo$remove]
testDescr<-testDescr[, -comboInfo$remove]

############################################################################
# Tuning training data set
############################################################################

#For the train function, the possible resampling methods are: bootstrapping, 
#k-fold cross- validation, leave-one-out cross-validation, and leave-group-
#out cross-validation (i.e., repeated splits without replacement).

#Arguments are:
#a matrix or data.frame of predictors - must be numeric
#a vector for outcomes - might be numeric or factors
#method used to train the dataset. To see options, check pg 9 - https://drive.google.com/open?id=0B4TReYGK49h_UU5wR2IwU3BMSzg
#a vector discussing the metrics to be returned. Options are "Accuracy", "Kappa", "RMSE" or "Rsquared"
#trControl: a list of control parameters such as number of resamples

#creating controlling variable
bootControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, allowParallel = TRUE, search = "random")
bootControl_rf <- trainControl(method = "oob", number = 5, allowParallel = TRUE, search = "random")
set.seed(2)

# building the tuning model with the random forest method
#Joao confere a metrica a ser usada - fiquei um pouco indeciso sobre qual usar: Accuracy, Kappa, RMSE ou ROC
rfFit <- train(trainDescr, trainClass,
                        method = "parRF", tuneLength = 5,
                        metric = "Accuracy",
                        trControl = bootControl_rf)
rfFit
rfFit$finalModel

# building the tuning model with the bagged trees method
bagTFit <- train(trainDescr, trainClass,
                          method = "treebag", metric = "Accuracy", trControl = bootControl, tuneLength = 20)
bagTFit
bagTFit$finalModel

# Example with the boosted tree method
# here we are controling for size of tree, iterations and learning rate
gbmGrid <- expand.grid(.interaction.depth = (1:5) * 2,
                       .n.trees = (1:10)*25, .shrinkage = .1, .n.minobsinnode= 5)
set.seed(2)
gbmFit <- train(trainDescr, trainClass,
                         method = "gbm", trControl = bootControl, verbose = FALSE,
                         bag.fraction = 0.5, tuneGrid = gbmGrid)
plot(gbmGrid)
plot(gbmFit, metric = "Kappa")
plot(gbmFit, plotType = "level")
resampleHist(gbmFit)
gbmFit


############################################################################
# Prediction de novas amostras após o modelo ter sido rodado
############################################################################
rf_pred <- predict (rfFit, trainClass); confusionMatrix(rf_pred, testClass)



############################################################################
# Prediction Performance
############################################################################
rfFit.varimp <- varImp(rfFit)
bagTFit.varimp <- varImp(bagTFit)
gbmFit.varimp <- varImp(gbmFit)

plot(rfFit.varimp, main = "Importance of all Variables for 'rf' model")
plot(bagTFit.varimp, main = "Importance of all Variables for 'bagged tree' model")
plot(gbmFit.varimp, main = "Importance of all Variables for 'gbm' model")
























