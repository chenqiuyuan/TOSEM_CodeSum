package tutorial;

import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;

class PrettyPrintStarter {
    public static void main(String[] args) {
        ClassOrInterfaceDeclaration myClass = new ClassOrInterfaceDeclaration();
        myClass.setComment(new JavadocComment("A very cool class!"));
        myClass.setName("MyClass");
        myClass.addField("String", "foo");
        System.out.println(myClass);
    }
}