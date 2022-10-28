'''Purpose:
    Given a function to identify a pattern, the candles to identify it on, and a conditional to determine whether it succeeded, return the amount of times it succeeded and optionally print the charts where the pattern is identified.

Arguments:

    patternIdentifier(function) - The function used to identify the pattern given the candles to run it on. Must return either True or False representing whether the pattern is present, a dictionary with information for the conditional, and the candles to run the conditional on.

    candles(pandas DataFrame) - The candles to search for the pattern on

    conditional(function) - The function used to test whether the pattern hit its stop loss, take profit, or neither given the candles to run it on. Return 0,1, or 2 to represent whether the pattern hit its stop loss, nothing, or its take profit respectively and a dictionary of arguments to pass to the showGraphs function.

    showGraphs(function) - The function to show the graph if the pattern is present given the candles to show it on and a dictionary of arguments from the conditional.'''
import pandas as pd
def check_strat(patternIdentifier:function,candles:pd.DataFrame,conditional:function,showGraphs:function):
    #Initialize variables to count the wins, losses, and misses
    wins = 0
    losses = 0
    misses = 0
    #Run the identifier on the candles and assign the variables necessary for the conditional function to run
    present,condArgDict,condCandles = patternIdentifier(candles)
    if present:
        wlm,graphArgDict = conditional(condCandles,condArgDict)
        match wlm:
            case 0: losses+=1
            case 1: misses+=1
            case 2: wins+=1
        showGraphs(candles,graphArgDict)
    return (wins,losses,misses)