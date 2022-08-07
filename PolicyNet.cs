using NumSharp;
using System;
using System.Data;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;

namespace Models
{
    /// <summary></summary>
    public class PolicyNet
    {
        int n_features = 9;
        int n_classes = 5;

        // For state var standardization
        NDArray means = np.array(new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f });
        NDArray stds = np.array(new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f });

        IVariableV1 h1, h2, h3, h4, h5, wout, b1, b2, b3, b4, b5, bout;
        IVariableV1[] trainable_variables;
        OptimizerV2 optimizer;
        Random rand;

        /// <summary></summary>
        public PolicyNet DeepCopy()
        {
            PolicyNet clonedNet = (PolicyNet)this.MemberwiseClone();
            clonedNet.h1 = tf.Variable(h1);
            clonedNet.h2 = tf.Variable(h2);
            clonedNet.h3 = tf.Variable(h3);
            clonedNet.h4 = tf.Variable(h4);
            clonedNet.h5 = tf.Variable(h5);
            clonedNet.wout = tf.Variable(wout);
            clonedNet.b1 = tf.Variable(b1);
            clonedNet.b2 = tf.Variable(b2);
            clonedNet.b3 = tf.Variable(b3);
            clonedNet.b4 = tf.Variable(b4);
            clonedNet.b5 = tf.Variable(b5);
            clonedNet.bout = tf.Variable(bout);
            clonedNet.means = (NDArray)means.Clone();
            clonedNet.stds = (NDArray)stds.Clone();

            return clonedNet;
        }

        /// <summary></summary>
        public PolicyNet(int[] n_hidden, Random rnd)
        {
            tf.enable_eager_execution();

            rand = rnd;
            var ini = tf.initializers.random_normal_initializer(seed: rand.Next());
            h1 = tf.Variable(ini.Apply(new InitializerArgs((n_features, n_hidden[0]))));
            h2 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[0], n_hidden[1]))));
            h3 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[1], n_hidden[2]))));
            h4 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[2], n_hidden[3]))));
            h5 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[3], n_hidden[4]))));
            wout = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[4], n_classes))));
            b1 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[0]))));
            b2 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[1]))));
            b3 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[2]))));
            b4 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[3]))));
            b5 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[4]))));
            bout = tf.Variable(ini.Apply(new InitializerArgs((n_classes))));

            trainable_variables = new IVariableV1[] { h1, h2, h3, h4, h5, wout,
                                                      b1, b2, b3, b4, b5, bout };
        }


        /// <summary></summary>
        public void Update(float learning_rate, int episode)
        {
            NDArray S = np.hstack(np.array(Program.stages).reshape(-1, 1),
                                  np.array(Program.lais).reshape(-1, 1),
                                  np.array(Program.esw0s).reshape(-1, 1),
                                  np.array(Program.esw1s).reshape(-1, 1),
                                  np.array(Program.esw2s).reshape(-1, 1),
                                  np.array(Program.esw3s).reshape(-1, 1),
                                  np.array(Program.esw4s).reshape(-1, 1),
                                  np.array(Program.cuIrrigs).reshape(-1, 1),
                                  np.array(Program.cuRains).reshape(-1, 1));

            NDArray meansOld = (NDArray)means.Clone();
            NDArray stdsOld = (NDArray)stds.Clone();

            float factor = new int[] { episode, 100 }.Min();
            for (int i = 0; i < n_features; i++)
            {
                var column = S[$":,{i}"];

                means[i] = (column.mean() + meansOld[i] * (factor - 1)) / factor; // new mean
                stds[i] = (column.std() + stdsOld[i] * (factor - 1)) / factor; // new std

                // standardised states
                S[$":,{i}"] = (column - means[i]);
                //// Divide it only if not too small
                if (stds[i] > 1e-8f)
                    S[$":,{i}"] = S[$":,{i}"] / stds[i];
            }

            // A for masking, i.e. multiplying probabilities for unchosen actions by 0
            var A = tf.one_hot(np.array(Program.actions), n_classes);

            // To make use of masking by A, G has identical column entries at each row
            int n_days = Program.days.Count; // Same size in all the state variables
            float g = Program.priceOut * Program.yield; // revenue
            var G = np.ones((n_days, n_classes), dtype: np.float32); // initialization
            for (int i = n_days - 1; i >= 0; i--) // start from the harvest day
            {
                int a = Program.actions[i];
                g -= Program.priceWater * Program.amounts[a]; // water cost
                G[i] = G[i] * g; // each row (i) has the identical entry (g)
            }

            // Gradient ascent
            optimizer = tf.optimizers.SGD(learning_rate);
            using (var tape = tf.GradientTape())
            {
                var PROBS = Predict(S);
                // Multiplying by A, each row has only 1 nonzero entry
                var loss = tf.reduce_sum(-tf.math.log(PROBS) * A * G);
                var gradients = tape.gradient(loss, trainable_variables);
                optimizer.apply_gradients(zip(gradients,
                                trainable_variables.Select(x => x as ResourceVariable)));
            }
        }


        /// <summary>Called by the IrrigationPolicy script</summary>
        public Tuple<int, float[]> Action(float stage, float fw, float lai, float[] esw, float cuIrrig, float cuRain)
        {
            var x = new NDArray(new float[] { stage, lai, esw[0], esw[1], esw[2], esw[3], esw[4], cuIrrig, cuRain });

            // Standardization
            for (int i = 0; i < n_features; i++)
            {
                x[i] = x[i] - means[i];
                // Divide it only if not too small
                if (stds[i] > 1e-8f)
                    x[i] = x[i] / stds[i];
            }

            float[] probs = Predict(x.reshape(1, n_features)).ToArray<float>();
            float r = (float)rand.NextDouble();
            float sum = 0f;
            for (int a = 0; a < n_classes; a++)
            {
                sum += probs[a];
                if (sum >= r)
                    return Tuple.Create(a, probs);
            }
            // This is never reached but the compiler requires.
            return Tuple.Create(n_classes - 1, probs);
        }


        /// <summary></summary>
        public Tensor Predict(Tensor x)
        {
            var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
            layer_1 = tf.nn.relu(layer_1);
            var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
            layer_2 = tf.nn.relu(layer_2);
            var layer_3 = tf.add(tf.matmul(layer_2, h3.AsTensor()), b3.AsTensor());
            layer_3 = tf.nn.relu(layer_3);
            var layer_4 = tf.add(tf.matmul(layer_3, h4.AsTensor()), b4.AsTensor());
            layer_4 = tf.nn.relu(layer_4);
            var layer_5 = tf.add(tf.matmul(layer_4, h5.AsTensor()), b5.AsTensor());
            layer_5 = tf.nn.relu(layer_5);
            var out_layer = tf.matmul(layer_5, wout.AsTensor()) + bout.AsTensor();
            return tf.nn.softmax(out_layer);
        }
    }
}
