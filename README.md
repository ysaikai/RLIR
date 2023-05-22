# Deep reinforcement learning for irrigation scheduling using high-dimensional sensor feedback

by [Yuji Saikai](https://yujisaikai.com), Allan Peake, and [Karine Chenu](https://researchers.uq.edu.au/researcher/1740)

A preprint is on [arXiv](https://arxiv.org/abs/2301.00899). To capture a crop yield directly in the code, rather than from the output file (which is slow), the following lines were inserted in [L.228](https://github.com/APSIMInitiative/ApsimX/blob/4b7f31d2df86b0222ba9b796bf0ede40fa25a0c9/Models/Report/Report.cs#LL227C28-L227C28) in `Models/Report/Report.cs`.

`if (columns[i].Name == "Yield")`<br />
`    Program.yield = (float)Convert.ToDouble(valuesToWrite.Last());`

which assumes `Yield` as a reporting variable defined in `.apsimx` file. For example, `[Wheat].Grain.Total.Wt*10 as Yield`.

&nbsp;

**Abstract**<br />
Deep reinforcement learning has considerable potential to improve irrigation scheduling in many cropping systems by applying adaptive amounts of water based on various measurements over time. The goal is to discover an intelligent decision rule that processes information available to growers and prescribes sensible irrigation amounts for the time steps considered. Due to the technical novelty, however, the research on the technique remains sparse and impractical. To accelerate the progress, the paper proposes a principled framework and actionable procedure that allow researchers to formulate their own optimisation problems and implement solution algorithms based on deep reinforcement learning. The effectiveness of the framework was demonstrated using a case study of irrigated wheat grown in a productive region of Australia where profits were maximised. Specifically, the decision rule takes nine state variable inputs: crop phenological stage, leaf area index, extractable soil water for each of the five top layers, cumulative rainfall and cumulative irrigation. It returns a probabilistic prescription over five candidate irrigation amounts (0, 10, 20, 30 and 40 mm) every day. The production system was simulated at Goondiwindi using the APSIM-Wheat crop model. After training in the learning environment using 1981-2010 weather data, the learned decision rule was tested individually for each year of 2011-2020. The results were compared against the benchmark profits obtained by a conventional rule common in the region. The discovered decision rule prescribed daily irrigation amounts that uniformly improved on the conventional rule for all the testing years, and the largest improvement reached 17% in 2018. The framework is general and applicable to a wide range of cropping systems with realistic optimisation problems.


&nbsp;

**Prescribed daily irrigation probabilities**

![](probabilities.png)
![](probabilities2.png)

Dots (●) and triangles (▼) represent daily rainfall and realised irrigation amounts respectively.
