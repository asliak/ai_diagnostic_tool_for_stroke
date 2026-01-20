import 'package:flutter/material.dart';
import 'cerebrum_scanner.dart';
 
void main() {
  runApp(const CerebrumScannerApp());
}
 
class CerebrumScannerApp extends StatelessWidget {
  const CerebrumScannerApp({super.key});
 
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cerebrum Scanner',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        scaffoldBackgroundColor: const Color(0xFFF1F7F9),
      ),
      home: const CerebrumScannerPage(),
    );
  }
}