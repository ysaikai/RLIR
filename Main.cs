namespace Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.IO;
    using Models.Core.Run;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Linq;

    /// <summary>Class to hold a static main entry point.</summary>
    public class Program
    {
        /// <summary>policy network</summary>
        public static PolicyNet policyNet;
        /// <summary> A set of irrigation amounts </summary>
        public static float[] amounts = { 0f, 10f, 20f, 30f, 40f };
        /// <summary></summary>
        public static float yield;
        /// <summary></summary>
        public static float priceOut = 0.25f;
        /// <summary></summary>
        public static float priceWater = 0.6f;
        /// <summary></summary>
        public static int episode;
        /// <summary></summary>
        public static List<float> days;
        /// <summary></summary>
        public static List<float> stages;
        /// <summary></summary>
        public static List<float> fws;
        /// <summary></summary>
        public static List<float> lais;
        /// <summary></summary>
        public static List<float> esw0s;
        /// <summary></summary>
        public static List<float> esw1s;
        /// <summary></summary>
        public static List<float> esw2s;
        /// <summary></summary>
        public static List<float> esw3s;
        /// <summary></summary>
        public static List<float> esw4s;
        /// <summary></summary>
        public static List<float> esw5s;
        /// <summary></summary>
        public static List<float> esw6s;
        /// <summary></summary>
        public static List<float> cuIrrigs;
        /// <summary></summary>
        public static List<float> cuRains;
        /// <summary></summary>
        public static List<int> actions;
        /// <summary></summary>
        public static List<float[]> probs;

        static Dictionary<int, List<int>> actionsMax = new Dictionary<int, List<int>>();
        static Dictionary<int, float> profitMax = new Dictionary<int, float>();

        /// <summary></summary>
        public static void Main(string[] args)
        {
            // Parameters
            float lr;
            int n_episodes;
            int rndSeed;
            try // console inputs
            {
                lr = (float)Math.Pow(10, Int32.Parse(args[0]));
                n_episodes = Int32.Parse(args[1]);
                rndSeed = Int32.Parse(args[2]);
            }
            catch // default
            {
                lr = 1e-7f;
                n_episodes = 1000;
                rndSeed = (int)DateTime.Now.Ticks & 0x0000FFFF;
            }

            int order = 100; // Order of MA
            int[] n_hidden = { 400, 600, 800, 600, 400 };
            string location = "Goond";
            int yearStart = 1981;
            int yearEnd = 2020;
            var yearsExcluded = new List<int> { };

            int year;
            for (year = yearStart; year <= yearEnd; year++)
            {
                if (!yearsExcluded.Contains(year))
                {
                    profitMax.Add(year, 0f);
                    actionsMax.Add(year, new List<int>());
                }
            }

            // Variables
            float profit;
            float ma = 0f; // moving average
            float maMax = float.NegativeInfinity;
            int episodeMax = 0;
            Random rand = new Random(rndSeed);
            policyNet = new PolicyNet(n_hidden, rand);
            PolicyNet policyNetMax = new PolicyNet(n_hidden, rand);

            // Relative path
            string path = System.AppDomain.CurrentDomain.BaseDirectory;
            path = System.IO.Path.GetDirectoryName(path);
            path = System.IO.Path.GetDirectoryName(path); // two-level above

            string fileNameAPSIM = path + $"\\Wheat.apsimx";

            // Initial message
            string msg = "\n";
            msg += "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~\n";
            msg += "{0} episodes; lr={1}; seed={2}; H1-5={3}-{4}-{5}-{6}-{7}\n";
            msg += "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~";
            Console.WriteLine(msg, n_episodes, lr, rndSeed, n_hidden[0],
                              n_hidden[1], n_hidden[2], n_hidden[3], n_hidden[4]);

            //**********
            // Learning
            //**********
            var stopWatch = new System.Diagnostics.Stopwatch();
            for (episode = 1; episode <= n_episodes; episode++)
            {
                stopWatch.Start();
                Initialize();

                // Except the last 10 years
                do
                    year = rand.Next(yearStart, yearEnd - 9);
                while (yearsExcluded.Contains(year));

                ProcessApsimFile(fileNameAPSIM, year, location);
                var runner = new Runner(fileNameAPSIM);
                runner.Run();

                // Update
                profit = priceOut * yield - priceWater * cuIrrigs[days.Count - 1];
                if (profit > profitMax[year])
                {
                    profitMax[year] = profit;
                    actionsMax[year] = actions;
                }

                float factor = new int[] { episode, order }.Min();
                ma = (ma * (factor - 1) + profit) / factor;
                if (ma > maMax)
                {
                    maMax = ma;
                    episodeMax = episode;
                    policyNetMax = policyNet.DeepCopy(); // Save the best policy
                }
                policyNet.Update(lr, episode);

                // Progress
                if (episode % 500 == 0)
                {
                    stopWatch.Stop();
                    TimeSpan ts = stopWatch.Elapsed;
                    Console.WriteLine("{0,5:D1}: {1,4:F0} / {2:0} (at {3}) {4:00}:{5:00}",
                                      episode, ma, maMax, episodeMax, ts.Hours, ts.Minutes);
                }
            }

            //**********
            // Testing
            //**********
            policyNet = policyNetMax; // Demonstrate the best policy

            string dirName = path + $"\\re{DateTime.Now.ToString("yyyyMMddHHmmss")}";
            if (!Directory.Exists(dirName)) Directory.CreateDirectory(dirName);

            // Include the training years
            for (year = yearStart; year <= yearEnd; year++)
            {
                for (int j = 0; j < 30; j++)
                {
                    Initialize();
                    ProcessApsimFile(fileNameAPSIM, year, location);
                    var runner = new Runner(fileNameAPSIM);
                    runner.Run();

                    profit = priceOut * yield - priceWater * cuIrrigs[days.Count - 1];
                    using (var file = new StreamWriter(dirName + $"\\{year}_{j}.csv"))
                    {
                        file.WriteLine($"{profit}");
                        file.WriteLine($"{yield}");
                        file.WriteLine($"{cuIrrigs[days.Count - 1]}");
                        file.WriteLine($"{rndSeed}");
                        file.WriteLine($"{n_episodes}");
                        file.WriteLine($"{n_hidden[0]}-{n_hidden[1]}-{n_hidden[2]}-{n_hidden[3]}-{n_hidden[4]}");
                        file.WriteLine("");
                        file.WriteLine("Day, Stage,   FW,  LAI, ESW0, ESW1, ESW2, ESW3, ESW4, ESW5, ESW6, " +
                                       "cuIrrig, cuRain, Action, p(0), p(1), p(2), p(3), p(4)");
                        for (int i = 0; i < days.Count; i++)
                        {
                            file.Write("{0:0}, {1,5:F1}, {2,4:F2}, {3,4:F2}, {4,4:F0}, {5,4:F0}, {6,4:F0}, " +
                                       "{7,4:F0}, {8,4:F0}, {9,4:F0}, {10,4:F0}, {11,7:F0}, {12,6:F0}, {13,6:D1}, ",
                                           days[i],
                                           stages[i],
                                           fws[i],
                                           lais[i],
                                           esw0s[i],
                                           esw1s[i],
                                           esw2s[i],
                                           esw3s[i],
                                           esw4s[i],
                                           esw5s[i],
                                           esw6s[i],
                                           cuIrrigs[i],
                                           cuRains[i],
                                           actions[i]);
                            file.WriteLine("{0:0.00}, {1:0.00}, {2:0.00}, {3:0.00}, {4:0.00}",
                                           probs[i][0],
                                           probs[i][1],
                                           probs[i][2],
                                           probs[i][3],
                                           probs[i][4]);
                        }
                    }
                }
            }
        }


        static void Initialize()
        {
            yield = 0f;
            days = new List<float>();
            stages = new List<float>();
            fws = new List<float>();
            lais = new List<float>();
            esw0s = new List<float>();
            esw1s = new List<float>();
            esw2s = new List<float>();
            esw3s = new List<float>();
            esw4s = new List<float>();
            esw5s = new List<float>();
            esw6s = new List<float>();
            cuIrrigs = new List<float>();
            cuRains = new List<float>();
            actions = new List<int>();
            probs = new List<float[]>();
        }


        // Process .apsimx file as JSON file (https://stackoverflow.com/a/56027969)
        static void ProcessApsimFile(string fileNameAPSIM, int year, string location)
        {
            string jsonString = File.ReadAllText(fileNameAPSIM);
            JObject jObject = JsonConvert.DeserializeObject(jsonString) as JObject;
            JToken jtStart = jObject.SelectToken("Children[0].Children[0].Start");
            jtStart.Replace(year.ToString() + "-01-01T00:00:00");
            JToken jtEnd = jObject.SelectToken("Children[0].Children[0].End");
            jtEnd.Replace(year.ToString() + "-12-31T00:00:00");
            JToken jtLoc = jObject.SelectToken("Children[0].Children[2].FileName");
            jtLoc.Replace(".\\" + location + ".met");
            string updatedJsonString = jObject.ToString();
            File.WriteAllText(fileNameAPSIM, updatedJsonString);
        }
    }
}
