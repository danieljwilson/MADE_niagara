
library(tidyverse)

load("swap_v1_clean.RData")
load("swap_v2_clean.RData")

hist(swap_v1_clean$rt)
hist(swap_v2_clean$rt)

# Create fixations and expdata files
unique(swap_v1_clean$subject)
unique(swap_v2_clean$subject)

