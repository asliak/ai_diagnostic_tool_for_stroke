import 'dart:convert';
import 'dart:typed_data'; // Uint8List için gerekli
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class CerebrumScannerPage extends StatefulWidget {
  const CerebrumScannerPage({super.key});

  @override
  State<CerebrumScannerPage> createState() => _CerebrumScannerPageState();
}

class _CerebrumScannerPageState extends State<CerebrumScannerPage> {
  final String backend = "http://cerebrumscanner.com:44155";
  final TextEditingController _patientIdController = TextEditingController();
  final TransformationController _transformationController = TransformationController();

  bool _isLoading = false;
  String _prediction = "Waiting...";
  String _statusText = "Idle";
  Color _statusColor = const Color(0xFFD8E9F0);

  List<dynamic> _slices = [];
  // Görüntüleri byte olarak saklayacağımız liste (Performans için kritik)
  List<Uint8List> _cachedImages = []; 
  
  double _currentSliceIndex = 0;
  double _windowWidth = 1.0; 
  double _windowLevel = 0.0; 

  List<double> _getManualMatrix(double ww, double wl) {
    double brightness = wl * 100; 
    return [
      ww, 0, 0, 0, brightness,
      0, ww, 0, 0, brightness,
      0, 0, ww, 0, brightness,
      0, 0, 0, 1, 0,
    ];
  }

  Future<void> fetchLatest() async {
    final patientId = _patientIdController.text.trim();
    if (patientId.isEmpty) {
      setState(() {
        _statusText = "Please enter Patient ID";
        _statusColor = Colors.redAccent;
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _statusText = "Fetching & Decoding...";
      _cachedImages = []; // Listeyi temizle
    });

    try {
      final url = Uri.parse("$backend/get_latest_classification/$patientId");
      final res = await http.get(url).timeout(const Duration(seconds: 180));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final List<dynamic> incomingSlices = data["slices"] ?? [];

        // Görüntüleri arka planda hızlıca decode et
        List<Uint8List> decodedList = [];
        for (var slice in incomingSlices) {
          if (slice["image_base64"] != null) {
            decodedList.add(base64Decode(slice["image_base64"]));
          }
        }

        setState(() {
          _slices = incomingSlices;
          _cachedImages = decodedList; // Artık byte listemiz hazır
          _currentSliceIndex = 0;
          if (_slices.isNotEmpty) {
            _prediction = _slices[0]["prediction"]?.toString().toUpperCase() ?? "UNKNOWN";
            _statusText = "Loaded: ${_slices.length} slices";
            _statusColor = const Color(0xFF2ECC71);
          }
        });
      }
    } catch (e) {
      setState(() {
        _statusText = "Error occurred";
        _statusColor = Colors.redAccent;
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    // hasData kontrolünü cachedImages üzerinden yapıyoruz
    final bool hasData = _cachedImages.isNotEmpty && _currentSliceIndex < _cachedImages.length;
    
    return Scaffold(
      backgroundColor: const Color(0xFFF1F7F9),
      appBar: AppBar(title: const Text("Cerebrum Scanner"), centerTitle: true),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Column(
          children: [
            TextField(
              controller: _patientIdController,
              decoration: const InputDecoration(labelText: "Patient ID", border: OutlineInputBorder(), isDense: true),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(
                  child: Container(
                    height: 320,
                    decoration: BoxDecoration(color: Colors.black, borderRadius: BorderRadius.circular(15)),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(13),
                      child: Stack(
                        children: [
                          if (hasData)
                            InteractiveViewer(
                              transformationController: _transformationController,
                              child: Center(
                                child: ColorFiltered(
                                  colorFilter: ColorFilter.matrix(_getManualMatrix(_windowWidth, _windowLevel)),
                                  // Image.memory artık önbellekteki byte'ı anlık okur
                                  child: Image.memory(
                                    _cachedImages[_currentSliceIndex.toInt()], 
                                    fit: BoxFit.contain,
                                    gaplessPlayback: true, // Kesitler arası geçişte beyazlamayı önler
                                  ),
                                ),
                              ),
                            ),
                          if (hasData)
                            Positioned(
                              top: 10, left: 10,
                              child: Container(
                                padding: const EdgeInsets.all(5),
                                color: Colors.black54,
                                child: Text("SLICE: ${(_currentSliceIndex + 1).toInt()}/${_slices.length}", style: const TextStyle(color: Colors.cyanAccent, fontSize: 10)),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
                if (hasData) const SizedBox(width: 8),
                if (hasData)
                  SizedBox(
                    height: 320, width: 25,
                    child: RotatedBox(
                      quarterTurns: 1,
                      child: Slider(
                        value: _currentSliceIndex,
                        min: 0, max: (_slices.length - 1).toDouble(),
                        onChanged: (v) => setState(() {
                          _currentSliceIndex = v;
                          _prediction = _slices[v.toInt()]["prediction"]?.toString().toUpperCase() ?? "UNKNOWN";
                        }),
                      ),
                    ),
                  ),
              ],
            ),

            const SizedBox(height: 10),
            if (hasData)
              Container(
                padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 15),
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(8), border: Border.all(color: const Color(0xFF1F7AC7))),
                child: Text("PREDICTION: $_prediction", style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1F7AC7))),
              ),

            const SizedBox(height: 15),
            if (hasData) ...[
              const Text("Window Width (Contrast)", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
              Slider(value: _windowWidth, min: 0.1, max: 3.0, onChanged: (v) => setState(() => _windowWidth = v)),
              const Text("Window Level (Brightness)", style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
              Slider(value: _windowLevel, min: -1.0, max: 1.0, onChanged: (v) => setState(() => _windowLevel = v)),
            ],

            const SizedBox(height: 10),
            Container(
              height: 40, width: double.infinity,
              alignment: Alignment.center,
              decoration: BoxDecoration(color: _statusColor, borderRadius: BorderRadius.circular(10)),
              child: Text(_statusText, style: const TextStyle(color: Colors.white, fontSize: 13)),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isLoading ? null : fetchLatest,
                style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF1F7AC7), padding: const EdgeInsets.symmetric(vertical: 12)),
                child: Text(_isLoading ? "Decoding..." : "Get Results", style: const TextStyle(color: Colors.white)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}