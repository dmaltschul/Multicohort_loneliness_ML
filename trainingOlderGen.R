### Older generation training analysis

## N.B. Looking at interactions and deeper details of the XGBMs requires an
## external program "xgbfi" built on the same foundations as "xgboost".


library(xgboost)
library(OpenMPController)
library(Matrix)
library(DiagrammeR)
library(ggplot2)
library(MASS)
library(caret)
library(pdp)
library(DescTools)



### xgboost

vars = c(
  'sex','socialclass','education',
  'nochildren',
  'ipip20e','ipip20a','ipip20c','ipip20es','ipip20i',
  'g',
  'single','widowed','divorced','self_rated_health',
  'liveAlone','livewChild','livewFamily','livewOther')
mat.36 = as.matrix(df.36[!is.na(df.36$lonely),vars])
response = df.36$lonely[!is.na(df.36$lonely)]



### What objective (link) to use?
mean(as.integer(response)-1, na.rm=TRUE) # 0.849
var(as.integer(response), na.rm=TRUE) # 0.953
## Close enough for Poisson.



param = list(booster='gbtree',mx_depth=7, objective='count:poisson', nthread=3, silent=0)


set.seed(1234567)
start.time <- Sys.time()
thirtysix.gbm = train(x = mat.36,
                  y = as.integer(response)-1,
                  method = "xgbTree", 
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 10
                  , objective='count:poisson', nthread=3, booster='gbtree'
                  )
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # Time difference of 1.186992 hours


print(thirtysix.gbm$bestTune)
max(thirtysix.gbm$results$Rsquared)
min(thirtysix.gbm$results$RMSE)
## R^2 and RMSE get the same parameters
thirtysix.gbm$results[thirtysix.gbm$results$Rsquared==max(thirtysix.gbm$results$Rsquared),]



### Best parameters

## Poisson
set.seed(1234567)
thirtysix.xgb = xgboost(param, label = as.numeric(df.36$lonely[!is.na(df.36$lonely)])-1,
                        data = mat.36, verbose=0,
                        nrounds=50, eta=0.3, max_depth=2, gamma=0, colsample_bytree=0.6,
                        min_child_weight=1, subsample=0.8333333
                        ,objective='count:poisson'
)
importance.36 <- xgb.importance(feature_names=colnames(mat.36), model=thirtysix.xgb)
importance.36

g <- xgb.ggplot.importance(importance.36)
g + theme_bw()



## Partial dependance plots

pdp.es <- partial(thirtysix.gbm, pred.var = "ipip20es", plot = TRUE, rug = TRUE)
pdp.g <- partial(thirtysix.gbm, pred.var = "g", plot = TRUE, rug = TRUE)
pdp.es.g <- partial(thirtysix.gbm, pred.var = c("ipip20es", "g"),
                    plot = TRUE, chull = TRUE)
grid.arrange(pdp.es, pdp.g, pdp.es.g, ncol = 3)

## Tends to be better for continuous variables.

pdp.w <- partial(thirtysix.gbm, pred.var = "widowed", plot = TRUE, rug = TRUE)
pdp.lA <- partial(thirtysix.gbm, pred.var = "liveAlone", plot = TRUE, rug = TRUE)
pdp.w.lA <- partial(thirtysix.gbm, pred.var = c("widowed", "liveAlone"),
                    plot = TRUE, chull = TRUE)
grid.arrange(pdp.w, pdp.lA, pdp.w.lA, ncol = 3)


pdp.w <- partial(thirtysix.gbm, pred.var = "widowed", plot = TRUE, rug = TRUE)
pdp.es <- partial(thirtysix.gbm, pred.var = "ipip20es", plot = TRUE, rug = TRUE)
pdp.w.es <- partial(thirtysix.gbm, pred.var = c("widowed", "ipip20es"),
                    plot = TRUE, chull = TRUE)
grid.arrange(pdp.w, pdp.es, pdp.w.es, ncol = 3)



### Exporting for interaction checking
featureList <- colnames(mat.36)
featureVector <- c() 
for (i in 1:length(featureList)) { 
  featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
}
write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
xgb.dump(model = thirtysix.xgb, fname = 'xgb.dump', fmap = "fmap.txt", with_stats = TRUE)



### Single variable ordinal model building

o.m.0 = polr(lonely ~ 1, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone')]),])
o.m.0$n

o.m.1 = polr(lonely ~ liveAlone, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone'
                                              , 'ipip20es' # comment in and out for comparing models with same data size
                                              )]),])
o.m.1$n
summary(o.m.1)

anova(o.m.0, o.m.1)


o.m.2 = polr(lonely ~ liveAlone + ipip20es, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es'
                                              , 'widowed' # comment in and out for comparing models with same data size
                                              )]),])
o.m.2$n
summary(o.m.2)

anova(o.m.1, o.m.2)


o.m.3 = polr(lonely ~ liveAlone + ipip20es + widowed, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed'
                                              , 'self_rated_health' # comment in and out for comparing models with same data size
                                              )]),])
o.m.3$n
summary(o.m.3)

anova(o.m.2, o.m.3)


o.m.4 = polr(lonely ~ liveAlone + ipip20es + widowed + self_rated_health, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health'
                                              , 'socialclass' # comment in and out for comparing models with same data size
                                              )]),])
o.m.4$n
summary(o.m.4)

anova(o.m.3, o.m.4)


o.m.5 = polr(lonely ~ liveAlone + ipip20es + widowed + self_rated_health + socialclass, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass'
                                              , 'ipip20a' # comment in and out for comparing models with same data size
                                              )]),])
o.m.5$n
summary(o.m.5)

anova(o.m.4, o.m.5)


o.m.6 = polr(lonely ~ liveAlone + ipip20es + widowed + self_rated_health + socialclass + ipip20a, 
           data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass','ipip20a')]),])
o.m.6$n
summary(o.m.6)

anova(o.m.5, o.m.6)
## stop at o.m.5



### Interaction depth 1

o.m1.1 = polr(lonely ~ liveAlone * ipip20es + widowed + self_rated_health + socialclass, 
            data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass')]),])
o.m1.1$n
summary(o.m1.1)

anova(o.m.5, o.m1.1)
## keep going


o.m1.2 = polr(lonely ~ liveAlone * ipip20es + liveAlone*sex + widowed + self_rated_health + socialclass, 
            data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass')]),])
o.m1.2$n
summary(o.m1.2)

anova(o.m1.1, o.m1.2)
## keep going


o.m1.3 = polr(lonely ~ liveAlone * ipip20es + liveAlone*sex + widowed + self_rated_health + socialclass + ipip20a*liveAlone 
            , data=df.36[complete.cases(df.36[,c('lonely','liveAlone','ipip20es','widowed','self_rated_health','socialclass')]),])
o.m1.3$n
summary(o.m1.3)

anova(o.m1.2, o.m1.3)
## Stop at m1.2.



### R^2 type measures

PseudoR2(o.m.5, which='Nagelkerke')

PseudoR2(o.m1.1, which='Nagelkerke')

PseudoR2(o.m1.2, which='Nagelkerke')


