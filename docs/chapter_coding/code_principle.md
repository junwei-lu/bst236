# Coding Principles




## Pivot your coding mindset

Here are the principles that you should pivot your coding mindset towards:

1. Don't repeat yourself: Learning how to automate and/or simplify repetitive tasks in your workflow because the time savings will be enormous.
2. Improve your coding style: Writing code that is more readable and maintainable before worrying about efficiency.
3. Leverage AI tools: Using AI copilot and other AI assistants to help you write better code faster while still understanding what the code does.
   
## Advices for coders



### **Understand code before relying on it.**

It's OK to __copy/paste code__ from Google or Stack Overflow. It's OK to ask AI to generate code for you. I myself am doing these all the time. However, make sure you __understand how it works__.  

Here are some ways to understand code:
* Run line by line and see what each does. 
* Change the code and see if it behaves as expected.
* Ask AI to explain the code with confidence level of 95% (please adjust the level based on your experience).



Here are some common pitfalls:
- **Beware of the context.** What works in one context may not work in another. Reading other's solution but we need to understand those phrases before we incorporate them into our project. When using AI tools, also make sure you incorporate the enought context (the entire codebase, the API documentation, the method description document,etc.) into your project.
- **AI hallucination.** AI is not always correct. In fact, it hallucinates a lot. So always try to verify the code.

However, you cannot guarantee you understand every detail of the code (like how the hardware works behind the code). Everything is about balance. I want to share a quote:

!!! note "Pablo Picasso:"
  
  Good artists copy, great artists steal.


And please think about what does "steal" meant by Picasso.
### Don't repeat yourself

Don't repeat yourself (DRY) is a fundamental concept in programming. 



- **DRY in Workflow:** Learning how to automate and/or simplify repetitive tasks in your workflow because the time savings will be enormous. We will cover some tools to do this in this course, but you first need to be aware of the existence of these tools. 
    * Best practice: Optimize your workflow from time to time but not too often.
    * Shell Script: Process files in batches, customize your commands in the system.
    * GNU Make: Create a `Makefile` to customize the project-specific build process. It is also a "cheat sheet" for common commands.
    * Transferable Workflows: Find the best practices making your workflow transferable across different platforms and machines.
    * Keyboard shortcuts: Use your keyboard more and your mouse less. 
- **DRY in Programming:** What if you need to change 1 thing? Needs to be changed in multiple places. Risk getting a wrong answer because we forgot to change one small thing.

    * Write code __a bit more general__ than your data or specific task.
      - Don't assume particular dimensions.
      - Don't forget about missing values (even if *your* data have none).
      - But **don't try to handle every case.** Try to anticipate what you might be asked for, but don't prepare for every possibility.
    * Use __function arguments__ to handle different cases. 
      - Don't assume particular file names.
      - Don't assume particular tuning parameters. 
      - Don't assume particular regression formulas.
      - No magic numbers.
      ``` r
      get_bootstrap_ci <- function(..., nboot = 1e3)
      ```
      ``` python
      def get_bootstrap_ci(..., nboot=1000):
      ```
    * Here are the tools to help you write more general code:
      - Python configuration files .
      - R `Renviron` file (See Workflow chapter).

### Think and document before you code

 You should always document what you want the code to do before you start writing it. With the help of AI tools, this documentation can not only help your thinking process but also will help AI tools to understand your goals. You could keep updating this documentation as you develop the code.

Here is an example of before your develop a package. 
```r
# Averaging over drtmle option:
#   Instantiate empty lists 
#   Note: length = 1 if n_SL = 1 or "drtmle" not in avg_over):
#     nuisance_drtmle 
#     nuisance_aiptw_c
#     ic_drtmle 
#     QnMod, gnMod, QrnMod, grnMod
#     drtmle -- eventually avg over to get final point estimates
#     aiptw_c -- eventually avg over 
#     tmle -- eventually avg over
#     aiptw -- eventually avg over
#     gcomp -- eventually avg over
#     validRows -- should this be added to the output?
#   Wrap everything into a big for loop and add indexes to the
#   objects above
#   Add code below for loop to do aggregation

# For introducing cross-validated standard errors: 
#   if se_cv = "full", but cvFolds = 1
#   need to modify make_validRows recognize this situation?
#   Or could modify estimateQ/g directly and have the functions
#   call themselves if se_cv = "full" and cvFolds == 1
#   or probably better to add an if statement that would call the 
#   estimateQ again outside. Yes, that's probably the way to go
```
This is copied from a local branch of `drtmle` R package from [David Benkeser](https://github.com/benkeser).

This document can also be used as the part of prompt engineering to help generative AI tools to understand your goals and help you write the code in the right direction.

Before you start coding, you can first "code" in natural language what you want to do.
You can write in the document on:
- The goal of the package
- How to decompose the goal into different functions
- The inputs and outputs of the functions you need.
- The potential approaches to implement the functions.

We call it "chain of thought" now in the context of using language models. We will talk about how to use AI tools to generate this meta-code but you need to be aware that you should think yourself. You should never be the slave of AI.





### **Premature optimization is the root of all evil.**

After reading all the advices above, you might be tempted to worry a lot before getting started with your coding. However, this will make me procrastinate and not get started forever. So how to balance?  My suggestion is to create a workflow template with all the best practices set up. Every time you start a new project, you just need to copy the template and modify it. This way you can focus on the content of your project and don't need to worry about the details of efficiency, reproducibility, etc. 

Based on the workflow template you have set up, now we can talk about the "vomit coding".
Here is the famous statement from [Kent Beck](https://en.wikipedia.org/wiki/Kent_Beck) for the order of importance of code:

- Make It Work 
- Make It Right 
- Make It Fast

So what is the difference between "work" and "right"? Here's one [interpretation](https://wiki.c2.com/?MakeItWorkMakeItRightMakeItFast): First crank out code that handles one common case. Then fix all of the special cases, error handling, etc. so all tests pass.

Remember: you don't get bonus points for code that "looks impressive".

  In Donald Knuth's paper "Structured Programming With GoTo Statements", he also wrote: "Programmers waste enormous amounts of time thinking about, or worrying about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered. We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%."




However, in this course, we will teach you a lot about how to make your code more efficient. Does that mean we violate the above principle? I want to answer this question by the another quote:

!!! note "Pablo Picasso:"
  
  It took me four years to paint like Raphael, but a lifetime to paint like a child.






Finding the right balance between readability and performance is a skill that develops with experience.