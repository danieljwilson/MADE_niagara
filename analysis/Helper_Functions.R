# HELPER FUNCTIONS


##################################
# Rename rData object
##################################

saveit <- function(..., file) {
  x <- list(...)
  save(list=names(x), file=file, envir=list2env(x))
}

## Usage
saveit(new_name=object, file="saved_file_name.RData")

##################################
##################################

