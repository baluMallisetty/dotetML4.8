using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace dotetML4._8
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class ModelOutput : ModelInput
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            //Step 1. Create an ML Context
            var ctx = new MLContext();

            //Step 2. Read in the input data from a text file for model training
            IDataView trainingData = ctx.Data
                .LoadFromTextFile<ModelInput>(args[0], hasHeader: false);

            //Step 3. Build your data processing and training pipeline
            var pipeline = ctx.Transforms.Text
                .FeaturizeText("Features", nameof(ModelInput.SentimentText))
                .Append(ctx.BinaryClassification.Trainers
                    .LbfgsLogisticRegression("Label", "Features"));

            //Step 4. Train your model
            ITransformer trainedModel = pipeline.Fit(trainingData);

            //Step 5. Make predictions using your trained model
            var predictionEngine = ctx.Model
                .CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            var sampleStatement = new ModelInput() { SentimentText = args[1] };

            var prediction = predictionEngine.Predict(sampleStatement);
            Console.WriteLine(Newtonsoft.Json.JsonConvert.SerializeObject(prediction));
           // Console.ReadKey();
        }
    }
}
