

data.table::setDTthreads(1)
devtools::document("autoxgboost3")

devtools::load_all("autoxgboost3")


library("mlr3")

task = tsk("iris")

autoxgboost_model(task, tune.threshold = FALSE)

