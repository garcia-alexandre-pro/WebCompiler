using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class IcedCoffeeScriptTest
    {
        private ConfigFileProcessor _processor;

        [TestInitialize]
        public void Setup()
        {
            _processor = new ConfigFileProcessor();
        }

        [TestCleanup]
        public void Cleanup()
        {
            File.Delete("../../artifacts/iced/test.js");
            File.Delete("../../artifacts/iced/test.min.js");
            File.Delete("../../artifacts/iced/test.js.map");
        }

        [TestMethod, TestCategory("Iced CoffeeScript")]
        public void CompileIcedCoffeeScript()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/icedcoffeeconfig.json");
            FileInfo js = new FileInfo("../../artifacts/iced/test.js");
            FileInfo map = new FileInfo("../../artifacts/iced/test.js.map");
            Assert.IsTrue(js.Exists);
            Assert.IsTrue(map.Exists);
            Assert.IsTrue(js.Length > 5);
            Assert.IsTrue(map.Length > 5);
        }

        [TestMethod, TestCategory("Iced CoffeeScript")]
        public void CompileIcedCoffeeScriptWithError()
        {
            IEnumerable<CompilerResult> result = _processor.Process("../../artifacts/icedcoffeeconfigerror.json");
            CompilerError error = result.First().Errors[0];
            Assert.AreEqual(1, error.LineNumber);
            Assert.AreEqual("unexpected ==", error.Message);
        }
    }
}
