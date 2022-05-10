using Microsoft.VisualStudio.TestTools.UnitTesting;
using WebCompiler;

namespace WebCompilerTest
{
    [TestClass]
    public class CompileServiceTest
    {
        [TestMethod, TestCategory("CompileService")]
        public void LessIsSupported()
        {
            bool result = CompilerService.IsSupported(".LESS");
            Assert.IsTrue(result);
        }

        [TestMethod, TestCategory("CompileService")]
        public void ScssIsSupported()
        {
            bool result = CompilerService.IsSupported(".SCSS");
            Assert.IsTrue(result);
        }

        [TestMethod, TestCategory("CompileService")]
        public void CoffeeIsSupported()
        {
            bool result = CompilerService.IsSupported(".COFFEE");
            Assert.IsTrue(result);
        }

        [TestMethod, TestCategory("CompileService")]
        public void HandleBarsIsSupported()
        {
            bool result = CompilerService.IsSupported(".HANDLEBARS");
            Assert.IsTrue(result);

            result = CompilerService.IsSupported(".hbs");
            Assert.IsTrue(result);
        }

        [TestMethod, TestCategory("CompileService")]
        public void LowerCaseSupportedExtensionAlsoWorks()
        {
            bool result = CompilerService.IsSupported(".less");
            Assert.IsTrue(result);
        }

        [TestMethod, TestCategory("CompileService")]
        public void OtherExtensionDoesntWorks()
        {
            bool result = CompilerService.IsSupported(".cs");
            Assert.IsFalse(result);
        }

    }
}
