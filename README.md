# Codebase for the research paper entitled "Optimal Control of Renewable Energy Communities subject to Network Peak Fees with Model Predictive Control and Reinforcement Learning Algorithms"

## In a nutshell

This codebase is the implementation of the optimal control framework for renewable energy communities (RECs) described in the research paper. In short (see the [European Union directives](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023L2413&qid=1699364355105) and the [recently adopted decree adopted by the Walloon Region ](https://www.ejustice.just.fgov.be/cgi/article_body.pl?language=fr&caller=summary&pub_date=2023-09-28&numac=2023044651) for more details), the surplus of electricity production of REC members can be shared to other members up to their demand, which in turn decrease the overall electricity bill. Cost savings are (or at least should be) redistributed among the members in some way, but this is out of the scope of this paper, which focuses on the minimisation of the sum of these REC electricity bills. In our setting, since we consider companies as members, these bills are composed of costs related to buying and selling electricity to retailers, local REC network fees, and peak demand (off-take and injection) costs. The latter is one of the prominent costs. In the shed of light of this observation, we propose that these peak costs should be (at least partially) computed from the energy bought and sold from the retailers. The rationale expressed in the paper is that we face two emergencies at the same time : global warming and risks of outage due to overloading electrical grids and centralised powerplants.

### Contributions:

- Description of the REC dynamics (cost function and state transitions for controllable assets such as batteries, and also other variables like meter readings)
- Modelisation of the REC decision process as a Partially Observable Markov Decision Process (POMDP)
- Benchmark of several algorithms : Model Predictive Control (repeated Mixed Integer Linear Program solves in a receding horizon), Reinforcement Learning (training parameterised, differentiable (w.r.t its parameters) functions with [Proximal Policy Optimisation](https://arxiv.org/abs/1707.06347), and rule-based policies that maximise some self-consumption criteria.
- Take-home messages
    - Peak costs are really prominent and should be taken into account. It strongly advocates for the peak costs to be computed only from retail exchanges, from the REC standpoint. Unfortunately, this is not happening in practice.
    - They also make our life harder : while optimal repartition of the energy produced can be computed with interior point methods, we did not identify any closed-form solution (without these peaks, a greedy algorithm is shown to be optimal since it can be implemented through a Fractional Knapsack Problem)
    - MPCs are quick to put in motion and to be adapted to other RECs. They also have the best performances (if accounting for peak costs) compared to other algos but remain dependant on the quality of the forecasted data, and are slow to execute (especially if MILP, binary variables -> difficulty scalable to large-sized RECs)
    - RL is very slow to train but once trained, but have shown to be able to have acceptable results once its trained, especially compared to its fast computation (something like 0.0001s).
    - Choosing between them is hence up to the needs (however we would strongly advise against a industry-grade product that is based on RL *only*)
    - Regardless the algorithm, controllable assets have a marginal effect on cost savings, even with rather optimistic sizing (and we did not consider investment costs...)

## Dependencies

- Install Python 3.8+;
- Run the command `python -m pip install -r requirements.txt`;


## Reproducing the results

TODO



## Roadmap to polish this code before (hopefully) the 1st January 2O24:

  - README
  - Clean implementation (removing redundant scripts, improve on modular patterns...)
  - Technical documentation of classes and functions 
  - Unit tests
  - Provide a Jupyter notebook example
