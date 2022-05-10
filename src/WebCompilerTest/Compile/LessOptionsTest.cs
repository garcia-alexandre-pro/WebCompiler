using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class LessOptionsTest
    {
        [TestMethod, TestCategory("LessOptions")]
        public void RelativeUrls()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/lessconfig.json");
            LessOptions result =  LessOptions.FromConfig(configs.First());
            Assert.AreEqual(true, result.RelativeUrls);
        }

        [TestMethod, TestCategory("LessOptions")]
        public void RootPath()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/lessconfig.json");
            LessOptions result = LessOptions.FromConfig(configs.First());
            Assert.AreEqual("./", result.RootPath);
        }

        [TestMethod, TestCategory("LessOptions")]
        public void StrictMath()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/lessconfig.json");
            LessOptions result = LessOptions.FromConfig(configs.First());
            Assert.AreEqual("strict", result.Math);
        }
    }
}
