library(GenSA)
library(DEoptim)
library(rgenoud)

nb.metrics <- 6
bench.run <- function(deltas=c( 1e-5, 1e-7, 1e-9),nbruns=100, benchfuns=NULL, meth="GenSA") {	
    nam <- NULL
    fun.res.list <- vector("list",length(benchfuns))
    computed_index <- 1
    for (fun.index in 1:length(benchfuns)) {
        nam <- c(nam,  benchfuns[[fun.index]]$name)
        cat(paste("================================\nProcessing", benchfuns[[fun.index]]$name, "function...\n================================\n"))
        print(benchfuns[[fun.index]]$fn)
        mat.method.list <- vector("list",0)
        ind <- 0
        for (delta in deltas) {
            cat(paste("Using delta: ", delta, '\n----------------------------------\n'))
            ind <- ind + 1
            mat.method <- matrix(NA, nbruns, 3)
            for(i in 1:nbruns) {
                #cat(paste("RUN #", i, '\n----------------------------------\n'))
                set.seed(1000 + i)
                nfev <<- 0
                firstHit <<- TRUE
                feval.suc <<- NA 
                fn.call.suc <<- NA
                TolF <<- benchfuns[[fun.index]]$glob.min + delta
                print(paste("Global min:", TolF))
                #cat(paste("glob.min to reach:", benchfuns[[fun.index]]$glob.min + delta, '\n'))
                if (meth=='GenSA') {
                    #print(packageVersion('GenSA'))
                    print(benchfuns[[fun.index]]$lower)
                    print(benchfuns[[fun.index]]$upper)
                    out <- GenSA(
                                 lower = benchfuns[[fun.index]]$lower,
                                 upper = benchfuns[[fun.index]]$upper,
                                 fn = benchfuns[[fun.index]]$fn,
                                 control=list(
                                              threshold.stop=benchfuns[[fun.index]]$glob.min + delta))
                    mat.method[i,] <- c(feval.suc, fn.call.suc, nfev)
                } else if ((meth=='DEoptim')){
                    #print(packageVersion('DEoptim'))
                    sink("/dev/null")
                    out <- DEoptim(lower = benchfuns[[fun.index]]$lower,
                                   upper = benchfuns[[fun.index]]$upper,
                                   fn = benchfuns[[fun.index]]$fn)
                    sink()
                    mat.method[i,] <- c(feval.suc, fn.call.suc, nfev)
                } else if (meth=='Rmalschains'){
                    sink("/dev/null")
                    out.malschains <- malschains(lower = benchfuns[[fun.index]]$lower,
                                                 upper = benchfuns[[fun.index]]$upper,
                                                 fn = benchfuns[[fun.index]]$fn, maxEvals=200000,
                                                 initialpop = seq(0.1, 0.1, length=30),
                                                 control=malschains.control(popsize=50,
                                                                            istep=300, ls="cmaes"))
                    sink()
                    mat.method[i,] <- c(feval.suc, fn.call.suc, nfev)
                } else if (meth=='rgenoud'){
                    domains <- matrix(c(benchfuns[[fun.index]]$lower,benchfuns[[fun.index]]$upper), ncol=2)
                    # rgenoud sometimes crash when calling stats:optim with infinite values
                    # So lets call it in a tryCatch block
                    out.rgenoud <- tryCatch({
                        sink("/dev/null")
                        out <- genoud(fn=benchfuns[[fun.index]]$fn, nvars=length(benchfuns[[fun.index]]$lower),
                                          Domains=domains)
                    }, error = function(err) {
                        print('Error when calling rgenoud:')
                        print(err)
                    }, finally = {
                        sink()
                        mat.method[i,] <- c(feval.suc, fn.call.suc, nfev)
                    })
                } else {
                    sink("/dev/null")
                    #print(packageVersion('DEoptim'))
                    out.DEoptim <- DEoptim(lower = benchfuns[[fun.index]]$lower,
                                           upper = benchfuns[[fun.index]]$upper,
                                           fn = benchfuns[[fun.index]]$fn)
                    sink()
                    out <- optim(par = out.DEoptim$optim$bestmem,
                                 fn = benchfuns[[fun.index]]$fn, lower=benchfuns[[fun.index]]$lower, upper=benchfuns[[fun.index]]$upper,
                                 method = "L-BFGS-B")
                    mat.method[i,] <- c(feval.suc, fn.call.suc, nfev)
                }
                cat(paste("(",benchfuns[[fun.index]]$name,":", benchfuns[[fun.index]]$dim,"Run#", i, "res:", feval.suc, fn.call.suc, nfev, "\n"))
                #if(any(c(out.GenSA$value, out.GenSA$counts, out.GenSA$counts) 
                # != mat.GenSA[i,])) 
                # warning("any(c(out.GenSA$value, out.GenSA$counts) != mat.GenSA[i,])")
            }
            colnames(mat.method) <- c("feval.suc", "feval.suc", "nfev")
            mat.method.list[[ind]] <- mat.method
        }
        fun.res.list[[computed_index]] <- mat.method.list
        computed_index <- computed_index + 1
    }
    names(fun.res.list) <- nam
    ind <- 1
    good.ind <- 1
    results <- vector("list",computed_index-1)
    for (name in names(fun.res.list)) {
        if (!is.null(fun.res.list[[ind]])) {
            names(results)[[good.ind]] <- name
            del.ind <- 0
            colnam <- NULL
            results[[good.ind]] <- vector("list", length(deltas))
            names(results[[good.ind]]) <- as.character(deltas)
            for (delta.ind in seq(from=1, to=length(deltas)*nb.metrics, by=nb.metrics)) {
                del.ind <- del.ind + 1
                results[[good.ind]][[del.ind]] <- vector("list", nb.metrics)
                names(results[[good.ind]][[del.ind]]) <- c("success", "aveSucFC", "aveFC", "se", "worst", "best")
                results[[good.ind]][[del.ind]]$success <- length(which(! is.na(fun.res.list[[ind]][[del.ind]][,2]))) / nbruns * 100 
                results[[good.ind]][[del.ind]]$aveSucFC <- mean(fun.res.list[[ind]][[del.ind]][,2],na.rm=TRUE)
                results[[good.ind]][[del.ind]]$aveFC <- mean(fun.res.list[[ind]][[del.ind]][,3],na.rm=TRUE)
                col.fc <- fun.res.list[[ind]][[del.ind]][,2]
                results[[good.ind]][[del.ind]]$se <- sd(col.fc,na.rm=TRUE) / sqrt(length(col.fc[which(!is.na(col.fc))]))
                results[[good.ind]][[del.ind]]$worst <- max(col.fc,na.rm=TRUE)
                results[[good.ind]][[del.ind]]$best <- min(col.fc,na.rm=TRUE)	
            }
            good.ind <- good.ind + 1
        }
        ind <- ind + 1
    }
    return(results)
}

save.res <- function(obj, filepath) {
    save(obj, file=filepath)
}
