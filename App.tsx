import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Button, StyleSheet, Text, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { detect, type Detection } from './src/yolo';

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const photoOutput = usePhotoOutput();
  const tflite = useTensorflowModel(require('./assets/coin_model.tflite'), []);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Detection[] | null>(null);
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const capture = async () => {
    if (tflite.state !== 'loaded') return;
    setBusy(true);
    setError(undefined);
    try {
      const photo = await photoOutput.capturePhoto({}, {});
      const image = await photo.toImageAsync();
      photo.dispose();
      setResults(await detect(tflite.model, image));
      image.dispose();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!hasPermission || device == null) {
    return (
      <View style={styles.center}>
        <Text>Waiting for camera permission...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera style={StyleSheet.absoluteFill} device={device} outputs={[photoOutput]} isActive />
      <View style={styles.panel}>
        {results && (
          <View>
            {results.length === 0 ? (
              <Text style={styles.count}>Nothing detected</Text>
            ) : (
              Object.entries(
                results.reduce<Record<string, number>>((acc, d) => {
                  acc[d.label] = (acc[d.label] ?? 0) + 1;
                  return acc;
                }, {}),
              ).map(([label, n]) => (
                <Text key={label} style={styles.count}>
                  {label}: {n}
                </Text>
              ))
            )}
          </View>
        )}
        {error && <Text style={styles.error}>{error}</Text>}
        {busy || tflite.state === 'loading' ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Button title="Capture" onPress={capture} disabled={tflite.state !== 'loaded'} />
        )}
        {tflite.state === 'error' && <Text style={styles.error}>Model failed to load</Text>}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  panel: {
    position: 'absolute', left: 0, right: 0, bottom: 0, padding: 16, gap: 8,
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  count: { color: '#fff', fontSize: 24, fontWeight: '700' },
  error: { color: '#ff6b6b' },
});
