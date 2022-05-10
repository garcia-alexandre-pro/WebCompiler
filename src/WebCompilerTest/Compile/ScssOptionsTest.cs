using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class ScssOptionsTest
    {
        [TestMethod, TestCategory("SCSSOptions")]
        public void AutoPrefix()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/options/scss/scssconfigautoprefix.json");
            SassOptions result = WebCompiler.SassOptions.FromConfig(configs.ElementAt(0));
            Assert.AreEqual("test", result.AutoPrefix);
        }

        [TestMethod, TestCategory("SCSSOptions")]
        public void LoadPaths()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/options/scss/scssconfigloadpaths.json");
            SassOptions result = WebCompiler.SassOptions.FromConfig(configs.ElementAt(0));
            CollectionAssert.AreEqual(new string[] { "/test/test.scss", "/test/test2.scss" }, result.LoadPaths);
        }

        [TestMethod, TestCategory("SCSSOptions")]
        public void StyleExpanded()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/options/scss/scssconfigexpanded.json");
            SassOptions result = WebCompiler.SassOptions.FromConfig(configs.ElementAt(0));
            Assert.AreEqual(SassStyle.Expanded, result.Style);
        }

        [TestMethod, TestCategory("SCSSOptions")]
        public void StyleCompressed()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/options/scss/scssconfigcompressed.json");
            SassOptions result = WebCompiler.SassOptions.FromConfig(configs.ElementAt(0));
            Assert.AreEqual(SassStyle.Compressed, result.Style);
        }

        [TestMethod, TestCategory("SCSSOptions")]
        public void Precision()
        {
            IEnumerable<WebCompiler.Config> configs = ConfigHandler.GetConfigs("../../artifacts/options/scss/scssconfigprecision.json");
            SassOptions result = WebCompiler.SassOptions.FromConfig(configs.ElementAt(0));
            Assert.AreEqual(3, result.Precision);
        }
    }
}
