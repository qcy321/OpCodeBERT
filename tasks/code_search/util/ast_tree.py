import ast
from abc import ABC, abstractmethod


def generate_func(node: ast.FunctionDef):
    """
    Generate function information object
    :param node: AST function node information
    :return:
    """
    return FunctionNode(node)


def generate_class(node: ast.ClassDef):
    """
    Generate class information object
    :param node: AST class node information
    :return:
    """
    body = ClassNode(node)
    for inner_node in node.body:
        if isinstance(inner_node, ast.FunctionDef):
            body.functions.append(generate_func(inner_node))
        elif isinstance(inner_node, ast.ClassDef):
            body.inner_classes.append(generate_class(inner_node))
    return body


class CommentRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Remove all Expr nodes from the function body if they are Str type (i.e., multi-line comments or docstrings)
        node.body = [
            n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Str))
        ]
        return self.generic_visit(node)


class FunctionNode:
    """
    Function information storage object
    """

    def __init__(self, node) -> None:
        docstring = ast.get_docstring(node)
        remover = CommentRemover()
        source_code = ast.unparse(remover.visit(node))
        lines = [line for line in source_code.splitlines() if line.strip()]
        self.name = node.name if node.name else ""
        self.docstring = docstring if docstring else ""
        self.source_code = "\n".join(lines)


class ClassNode:
    """
    Class information storage object
    """

    def __init__(self, node) -> None:
        self.name = node.name if node.name else ""
        self.docstring = ast.get_docstring(node)
        self.functions: list[FunctionNode] = []
        self.inner_classes: list[ClassNode] = []


class FileNode:
    """
    Python file information storage object
    """

    def __init__(self, name):
        self.name = name
        self.docstring = ""
        self.functions: list[FunctionNode] = []
        self.classes: list[ClassNode] = []


class ExtractNode(ABC):
    """
    Extract function information from parsed objects by overriding _extract_func_node to perform desired operations
    """

    @abstractmethod
    def _extract_func_node(self, class_name: str, functions: list[FunctionNode], obj: any):
        """
        Needs to implement related logic
        :param class_name: Class name, i.e., the class to which the functions belong
        :param functions: Function information, storing all function information within a class
        :param obj: Object used to store data
        :return:
        """
        pass

    def _extract_class_node(self, class_name: str, classes: list[ClassNode], obj: any):
        for cla in classes:
            self._extract_class_node(f"{class_name}.{cla.name}" if class_name != "" else cla.name, cla.inner_classes,
                                     obj)
            self._extract_func_node(f"{class_name}.{cla.name}" if class_name != "" else cla.name, cla.functions, obj)

    def _extract_file_node(self, file_node: FileNode, obj: any):
        if file_node.name != "":
            self._extract_func_node("", file_node.functions, obj)
            self._extract_class_node("", file_node.classes, obj)

    def extract(self, file_node: FileNode, obj: any):
        self._extract_file_node(file_node, obj)


class Visitor(ast.NodeVisitor):
    """
    Inherit from ast.NodeVisitor to automatically parse different types of nodes and implement different operations,
    This is a wrapper for the desired functionality and is not exposed externally.
    """

    def __init__(self, name):
        self.file = FileNode(name)

    def visit_Expr(self, node):
        # Ensure the node has a value and is of string type
        if isinstance(node.value, ast.Str):
            # Extract docstring
            self.file.docstring = node.value.s

    def visit_ClassDef(self, node):
        # Parse inner classes
        self.file.classes.append(generate_class(node))

    def visit_FunctionDef(self, node):
        self.file.functions.append(generate_func(node))


def parse_code(code_str, file_name) -> FileNode:
    """
    Parse a Python file and extract function information
    :param code_str:
    :param file_name:
    :return:
    """
    try:
        tree = ast.parse(code_str)
        file_visitor = Visitor(file_name)
        file_visitor.visit(tree)
        return file_visitor.file
    except Exception as e:
        return FileNode("")


def parse_func_code(code_str) -> FunctionNode:
    """
    Parse only the function
    :param code_str:
    :return:
    """
    try:
        tree = ast.parse(code_str)
        return generate_func(tree.body[0])
    except Exception as e:
        return None
