package tutorial;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.printer.lexicalpreservation.LexicalPreservingPrinter;

class LexicalPreservingComplete {
    public static void main(String[] args) {
        String code = "// Hey, this is a comment\n\n\n// Another one\n\nclass A { }";
        CompilationUnit cu = StaticJavaParser.parse(code);
        LexicalPreservingPrinter.setup(cu); // 可以保持格式
        ClassOrInterfaceDeclaration myClass = cu.getClassByName("A").get();
        
        
        myClass.setName("MyNewClassName");
        myClass.addModifier(Modifier.Keyword.PUBLIC);
        cu.setPackageDeclaration("org.javaparser.lexicalpreservation.examples");
        System.out.println(LexicalPreservingPrinter.print(cu));
    }
}