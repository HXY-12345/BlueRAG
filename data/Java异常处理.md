---
title: Java异常处理
date: 2025-12-11 20:08:08
tags: Java基础
categories: Java
index_img: /img/Java异常处理.png   
banner_img: /img/Java异常处理.png  
---

# Java异常处理

# 异常

**Java 异常类层次结构图概览**：

![image.png](image.png)

从图中可以看出`IOException` 继承链是这样的：`IOException` → `Exception` → `Throwable`

## **Checked Exception 和 Unchecked Exception 有什么区别？**

**Checked Exception** 即 受检查异常 ，Java 代码在编译过程中，如果受检查异常没有被 `catch`或者`throws` 关键字处理的话，就没办法通过编译。

除了`RuntimeException`及其子类以外，其他的`Exception`类及其子类都属于受检查异常 。常见的受检查异常有：IO 相关的异常、`ClassNotFoundException`、`SQLException`...。

**Unchecked Exception** 即 **不受检查异常** ，Java 代码在编译过程中 ，我们即使不处理不受检查异常也可以正常通过编译。

`RuntimeException` 及其子类都统称为非受检查异常，常见的有（建议记下来，日常开发中会经常用到）：

- `NullPointerException`(空指针错误)
- `IllegalArgumentException`(参数错误比如方法入参类型错误)
- `NumberFormatException`（字符串转换为数字格式错误，`IllegalArgumentException`的子类）
- `ArrayIndexOutOfBoundsException`（数组越界错误）
- `ClassCastException`（类型转换错误）
- `ArithmeticException`（算术错误）
- `SecurityException` （安全错误比如权限不够）
- `UnsupportedOperationException`(不支持的操作错误比如重复创建同一用户)
- ……

## **你更倾向于使用 Checked Exception 还是 Unchecked Exception？**

默认使用 Unchecked Exception，只在必要时才用 Checked Exception。

我们可以把 Unchecked Exception（比如 `NullPointerException`）看作是代码 Bug。对待 Bug，最好的方式是让它暴露出来然后去修复代码，而不是用 `try-catch` 去掩盖它。

一般来说，只在一种情况下使用 Checked Exception：当这个异常是业务逻辑的一部分，并且调用方必须处理它时。比如说，一个余额不足异常。这不是 bug，而是一个正常的业务分支，我需要用 Checked Exception 来强制调用者去处理这种情况，比如提示用户去充值。这样就能在保证关键业务逻辑完整性的同时，让代码尽可能保持简洁。

## Throwable 类常用方法有哪些？

- `String getMessage()`: 返回异常发生时的详细信息
- `String toString()`: 返回异常发生时的简要描述
- `String getLocalizedMessage()`: 返回异常对象的本地化信息。使用 `Throwable` 的子类覆盖这个方法，可以生成本地化信息。如果子类没有覆盖该方法，则该方法返回的信息与 `getMessage()`返回的结果相同
- `void printStackTrace()`: 在控制台上打印 `Throwable` 对象封装的异常信息

# 代码示例

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Objects;

public class ExceptionDemo {

    public static void main(String[] args) {
        System.out.println("==== 1) Checked Exception 示例（必须处理/声明） ====");
        demoChecked();

        System.out.println("\n==== 2) Unchecked Exception 示例（运行时异常，可不强制处理） ====");
        demoUnchecked();

        System.out.println("\n==== 3) try-catch-finally 行为演示（finally 总会执行） ====");
        demoTryCatchFinally();

        System.out.println("\n==== 4) Throwable 常用方法演示 ====");
        demoThrowableMethods();

        System.out.println("\n==== 5) throw / throws + 自定义异常（扩展） ====");
        demoThrowAndThrows();
    }

    // -------------------------
    // 1) Checked Exception：编译期强制你处理
    // -------------------------
    static void demoChecked() {
        // Checked Exception 典型：IOException（读文件时可能发生）
        // 这里用 try-with-resources 演示：会自动关闭资源（底层也是 finally 的语义）
        String path = "not_exist.txt";

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            System.out.println(br.readLine());
        } catch (IOException e) { // IOException 属于 Checked Exception
            System.out.println("捕获到 Checked Exception: " + e.getClass().getName());
            System.out.println("原因：文件不存在或 IO 失败");
        }
    }

    // -------------------------
    // 2) Unchecked Exception：RuntimeException 及其子类
    // -------------------------
    static void demoUnchecked() {
        try {
            int a = 1;
            int b = 0;
            int c = a / b; // ArithmeticException（RuntimeException 子类）
            System.out.println(c); // 不会执行到这里
        } catch (ArithmeticException e) {
            System.out.println("捕获到 Unchecked Exception: " + e.getClass().getSimpleName());
            System.out.println("原因：除数不能为 0");
        }

        // 另一个常见 Unchecked：NullPointerException
        try {
            String s = null;
            System.out.println(s.length()); // NPE
        } catch (NullPointerException e) {
            System.out.println("捕获到 Unchecked Exception: " + e.getClass().getSimpleName());
            System.out.println("原因：对象为 null 却调用方法/属性");
        }
    }

    // -------------------------
    // 3) try-catch-finally：资源清理/收尾
    // -------------------------
    static void demoTryCatchFinally() {
        try {
            System.out.println("try：开始做点事情...");
            if (System.currentTimeMillis() > 0) {
                throw new IllegalStateException("模拟业务状态不对"); // Unchecked
            }
            System.out.println("try：正常结束（这里不会到）");
        } catch (IllegalStateException e) {
            System.out.println("catch：捕获并处理异常：" + e.getMessage());
            // 这里可以做：记录日志、封装后再抛出、给用户提示等
        } finally {
            // finally：无论是否异常、是否 return（大多数情况）都会执行
            System.out.println("finally：做收尾工作（关闭资源/释放锁/回滚等）");
        }
    }

    // -------------------------
    // 4) Throwable 常用方法：getMessage / getCause / printStackTrace / toString / getStackTrace
    // -------------------------
    static void demoThrowableMethods() {
        try {
            // 构造一个“异常链”：外层异常包装内层异常
            try {
                Integer.parseInt("abc"); // NumberFormatException（Unchecked）
            } catch (NumberFormatException inner) {
                throw new IOException("外层包装：把解析失败当成 IO 层错误", inner); // Checked
            }
        } catch (Throwable t) { // 注意：catch Throwable 很少用，演示用
            System.out.println("t.getClass(): " + t.getClass().getName());
            System.out.println("t.getMessage(): " + t.getMessage());
            System.out.println("t.getCause(): " + (t.getCause() == null ? "null" : t.getCause().getClass().getName()));
            System.out.println("t.toString(): " + t.toString());

            System.out.println("\n--- t.printStackTrace()（打印堆栈，定位最常用） ---");
            t.printStackTrace(System.out);

            System.out.println("\n--- t.getStackTrace()（拿到堆栈数组，可自定义输出） ---");
            StackTraceElement[] stack = t.getStackTrace();
            System.out.println("堆栈深度: " + stack.length);
            System.out.println("堆栈第1行示例: " + (stack.length > 0 ? stack[0] : "none"));
        }
    }

    // -------------------------
    // 5) throw / throws + 自定义异常
    // -------------------------
    static void demoThrowAndThrows() {
        try {
            checkUserName(null); // 这里会抛出自定义 Checked 异常
        } catch (InvalidUserException e) {
            System.out.println("捕获到自定义 Checked 异常: " + e.getMessage());
        }

        try {
            // Objects.requireNonNull 常用来主动抛出 NPE（Unchecked）
            Objects.requireNonNull(null, "参数不能为空");
        } catch (NullPointerException e) {
            System.out.println("捕获到 requireNonNull 抛出的 NPE: " + e.getMessage());
        }
    }

    // 自定义 Checked Exception：继承 Exception（不是 RuntimeException）
    static class InvalidUserException extends Exception {
        public InvalidUserException(String message) {
            super(message);
        }
    }

    // throws：声明该方法可能抛出某个 Checked Exception，让调用方决定如何处理
    static void checkUserName(String name) throws InvalidUserException {
        if (name == null) {
            // throw：主动抛异常（通常用于参数校验/业务校验失败）
            throw new InvalidUserException("用户名不能为空");
        }
    }
}
```

输出“

```java
==== 1) Checked Exception 示例（必须处理/声明） ====
捕获到 Checked Exception: java.io.FileNotFoundException
原因：文件不存在或 IO 失败

==== 2) Unchecked Exception 示例（运行时异常，可不强制处理） ====
捕获到 Unchecked Exception: ArithmeticException
原因：除数不能为 0
捕获到 Unchecked Exception: NullPointerException
原因：对象为 null 却调用方法/属性

==== 3) try-catch-finally 行为演示（finally 总会执行） ====
try：开始做点事情...
catch：捕获并处理异常：模拟业务状态不对
finally：做收尾工作（关闭资源/释放锁/回滚等）

==== 4) Throwable 常用方法演示 ====
t.getClass(): java.io.IOException
t.getMessage(): 外层包装：把解析失败当成 IO 层错误
t.getCause(): java.lang.NumberFormatException
t.toString(): java.io.IOException: 外层包装：把解析失败当成 IO 层错误

--- t.printStackTrace()（打印堆栈，定位最常用） ---
java.io.IOException: 外层包装：把解析失败当成 IO 层错误
	at org.hxxyy.bagu.ExceptionDemo.demoThrowableMethods(ExceptionDemo.java:94)
	at org.hxxyy.bagu.ExceptionDemo.main(ExceptionDemo.java:20)
Caused by: java.lang.NumberFormatException: For input string: "abc"
	at java.lang.NumberFormatException.forInputString(NumberFormatException.java:65)
	at java.lang.Integer.parseInt(Integer.java:580)
	at java.lang.Integer.parseInt(Integer.java:615)
	at org.hxxyy.bagu.ExceptionDemo.demoThrowableMethods(ExceptionDemo.java:92)
	... 1 more

--- t.getStackTrace()（拿到堆栈数组，可自定义输出） ---
堆栈深度: 2
堆栈第1行示例: org.hxxyy.bagu.ExceptionDemo.demoThrowableMethods(ExceptionDemo.java:94)

==== 5) throw / throws + 自定义异常（扩展） ====
捕获到自定义 Checked 异常: 用户名不能为空
捕获到 requireNonNull 抛出的 NPE: 参数不能为空

进程已结束，退出代码为 0
```

“必须处理/声明”，指的是：**编译器在编译期就要求你对某些异常给出交代**。以 `IOException` 为例，它属于 Checked Exception —— 代码里只要“可能抛出它”，Java 就不让你编译通过，除非你做了两种选择之一：

**1）“处理”：catch 掉（在当前方法里解决）**

```java
try (BufferedReaderbr=newBufferedReader(newFileReader(path))) {
    System.out.println(br.readLine());
}catch (IOException e) {
// 我在这里把异常处理了：提示、记录日志、兜底逻辑等
}
```

**2）“声明”：throws 抛给调用方（我不在这层处理）**

```java
staticvoiddemoChecked() throws IOException {
try (BufferedReaderbr=newBufferedReader(newFileReader("not_exist.txt"))) {
        System.out.println(br.readLine());
    }
}
```

## main方法抛出的异常谁来处理

```java
publicstaticvoidmain(String[] args)throws IOException { ... }
```

`main` 上写了 `throws IOException` 的意思是：**main 不处理这个 IOException，把它继续往外抛**。那问题来了：**外面是谁？**

结论：`main` 抛出去的异常，最终由 **JVM（Java 虚拟机）** 处理。

方法抛异常会沿着调用栈往外冒：

`demoChecked()` → `main()` → **JVM**

只要中间没有任何地方 catch 住它，它就会一直冒到最外层。`main` 已经是应用层最外层了，所以只能到 JVM。

## e.getMessage()是Throwable 类的方法

原因是：`IOException` 继承链是这样的：

`IOException` → `Exception` → `Throwable`

所以 `IOException e` 这个对象天然就拥有 `Throwable` 里定义的常用方法，包括：

- `e.getMessage()`：返回异常的“消息文本”（通常是构造异常时传入的 message）
- `e.getCause()`：返回根因异常（异常链）
- `e.printStackTrace()`：打印堆栈信息
- `e.toString()`：类名 + message
- `e.getStackTrace()`：堆栈数组

一个小提醒：`getMessage()` **可能返回 null**（如果创建异常时没传 message），而 `printStackTrace()` 基本都能给你最完整的定位信息。

# 参考

[Java基础常见面试题总结(下)](https://javaguide.cn/java/basis/java-basic-questions-03.html#%E5%BC%82%E5%B8%B8)