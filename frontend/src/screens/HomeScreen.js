import React, { useState, useEffect } from "react";
import {
    StyleSheet,
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Alert,
    KeyboardAvoidingView,
    Platform,
    SafeAreaView,
    Linking,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Clipboard from "expo-clipboard";

// Import components, services, and utilities
import { generateSummary } from "../services/api";
import {
    COLORS,
    SPACING,
    FONT_SIZES,
    SUMMARY_TYPES,
    SUMMARY_LENGTHS,
    SCREENS,
} from "../constants";

// Constants
const LAST_SETTINGS_KEY = "last_summary_settings";

const HomeScreen = ({ navigation, route }) => {
    // State
    const [url, setUrl] = useState("");
    const [isValidUrl, setIsValidUrl] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [summaryType, setSummaryType] = useState(SUMMARY_TYPES[0].id);
    const [summaryLength, setSummaryLength] = useState(SUMMARY_LENGTHS[1].id);

    // Function to handle shared text (URLs)
    const handleSharedText = async () => {
        try {
            // Check if app was opened from a share intent
            const initialUrl = await Linking.getInitialURL();
            if (initialUrl) {
                console.log("App opened from URL:", initialUrl);
                setUrl(initialUrl);

                // Process the URL immediately if it's a YouTube URL
                if (
                    initialUrl.includes("youtube.com/watch") ||
                    initialUrl.includes("youtu.be/") ||
                    initialUrl.includes("m.youtube.com/watch")
                ) {
                    // Use a timeout to ensure state is updated
                    setTimeout(() => {
                        processUrl(initialUrl);
                    }, 500);
                }
            }

            // For Android, we need to check for shared text
            if (Platform.OS === "android") {
                // This is a simplified approach - in a real app, you'd use
                // the Android native module to get the shared text
                console.log("Checking for Android shared text...");
                // Additional Android-specific handling can be added here
            }

            // For iOS, check for shared content
            if (Platform.OS === "ios") {
                console.log("Checking for iOS shared content...");
                // iOS-specific handling can be added here if needed
            }
        } catch (error) {
            console.error("Error handling shared text:", error);
        }
    };

    // Helper function to process a URL directly
    const processUrl = React.useCallback(
        async (urlToProcess) => {
            if (!urlToProcess || !urlToProcess.trim()) {
                return;
            }

            setIsLoading(true);
            try {
                console.log("Processing URL directly:", urlToProcess);
                const summary = await generateSummary(
                    urlToProcess,
                    summaryType,
                    summaryLength
                );
                navigation.navigate(SCREENS.SUMMARY, { summary });
            } catch (error) {
                console.error("Error processing URL:", error);
                Alert.alert(
                    "Error",
                    "Failed to process the URL. Please try again."
                );
            } finally {
                setIsLoading(false);
            }
        },
        [summaryType, summaryLength, navigation]
    );

    // Load last used settings and check for shared content
    useEffect(() => {
        const loadLastSettings = async () => {
            try {
                const settingsString = await AsyncStorage.getItem(
                    LAST_SETTINGS_KEY
                );
                if (settingsString) {
                    const settings = JSON.parse(settingsString);
                    setSummaryType(settings.type || SUMMARY_TYPES[0].id);
                    setSummaryLength(settings.length || SUMMARY_LENGTHS[1].id);
                }
            } catch (error) {
                console.error("Error loading last settings:", error);
            }
        };

        loadLastSettings();
        handleSharedText(); // Check for shared content when component mounts
    }, []);

    // Save settings when changed
    useEffect(() => {
        const saveSettings = async () => {
            try {
                await AsyncStorage.setItem(
                    LAST_SETTINGS_KEY,
                    JSON.stringify({ type: summaryType, length: summaryLength })
                );
            } catch (error) {
                console.error("Error saving settings:", error);
            }
        };

        saveSettings();
    }, [summaryType, summaryLength]);

    // Handle URL input change
    const handleUrlChange = (text) => {
        setUrl(text);
        setIsValidUrl(true); // Reset validation on change
    };

    // Handle clipboard paste
    const handlePasteFromClipboard = async () => {
        try {
            const clipboardContent = await Clipboard.getStringAsync();
            if (clipboardContent) {
                setUrl(clipboardContent);
                setIsValidUrl(true);
            }
        } catch (error) {
            console.error("Error pasting from clipboard:", error);
            Alert.alert("Error", "Failed to paste from clipboard");
        }
    };

    // Handle URL submission from the UI
    const handleSubmit = () => {
        // Basic validation - just check if URL is not empty
        if (!url.trim()) {
            setIsValidUrl(false);
            return;
        }

        // Use the common processUrl function
        processUrl(url);
    };

    // Handle shared URLs from navigation params
    useEffect(() => {
        if (route.params?.sharedUrl) {
            console.log(
                "Received shared URL from navigation:",
                route.params.sharedUrl
            );
            const sharedUrl = route.params.sharedUrl;
            setUrl(sharedUrl);

            // Automatically process the shared URL after a short delay
            // This ensures the URL is set in state before submission
            const timer = setTimeout(() => {
                console.log("Auto-processing shared URL:", sharedUrl);
                if (sharedUrl) {
                    processUrl(sharedUrl);
                }
            }, 500);

            return () => clearTimeout(timer);
        }
    }, [route.params?.sharedUrl, processUrl]);

    // Render summary type options
    const renderSummaryTypeOptions = () => {
        return (
            <View style={styles.optionsContainer}>
                <Text style={styles.optionsLabel}>Summary Type:</Text>
                <View style={styles.optionsButtonGroup}>
                    {SUMMARY_TYPES.map((type) => (
                        <TouchableOpacity
                            key={type.id}
                            style={[
                                styles.optionButton,
                                summaryType === type.id &&
                                    styles.optionButtonSelected,
                            ]}
                            onPress={() => setSummaryType(type.id)}
                        >
                            <Text
                                style={[
                                    styles.optionButtonText,
                                    summaryType === type.id &&
                                        styles.optionButtonTextSelected,
                                ]}
                            >
                                {type.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </View>
        );
    };

    // Render summary length options
    const renderSummaryLengthOptions = () => {
        return (
            <View style={styles.optionsContainer}>
                <Text style={styles.optionsLabel}>Summary Length:</Text>
                <View style={styles.optionsButtonGroup}>
                    {SUMMARY_LENGTHS.map((length) => (
                        <TouchableOpacity
                            key={length.id}
                            style={[
                                styles.optionButton,
                                summaryLength === length.id &&
                                    styles.optionButtonSelected,
                            ]}
                            onPress={() => setSummaryLength(length.id)}
                        >
                            <Text
                                style={[
                                    styles.optionButtonText,
                                    summaryLength === length.id &&
                                        styles.optionButtonTextSelected,
                                ]}
                            >
                                {length.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            </View>
        );
    };

    return (
        <SafeAreaView style={styles.container}>
            <KeyboardAvoidingView
                style={styles.container}
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={Platform.OS === "ios" ? 64 : 0}
            >
                <StatusBar style="auto" />
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    keyboardShouldPersistTaps="handled"
                >
                    <View style={styles.header}>
                        <Text style={styles.title}>YouTube Summarizer</Text>
                        <Text style={styles.subtitle}>
                            Get AI-powered summaries of YouTube videos
                        </Text>
                    </View>

                    <View style={styles.inputContainer}>
                        <View style={styles.inputRow}>
                            <TextInput
                                style={[
                                    styles.input,
                                    !isValidUrl && styles.inputError,
                                ]}
                                placeholder="Paste YouTube URL here"
                                value={url}
                                onChangeText={handleUrlChange}
                                autoCapitalize="none"
                                autoCorrect={false}
                                keyboardType="url"
                            />
                            <TouchableOpacity
                                style={styles.pasteButton}
                                onPress={handlePasteFromClipboard}
                                accessibilityLabel="Paste from clipboard"
                                accessibilityHint="Pastes YouTube URL from clipboard"
                            >
                                <Ionicons
                                    name="clipboard-outline"
                                    size={24}
                                    color={COLORS.primary}
                                />
                            </TouchableOpacity>
                        </View>
                        {!isValidUrl && (
                            <Text style={styles.errorText}>
                                Please enter a valid YouTube URL
                            </Text>
                        )}
                    </View>

                    {renderSummaryTypeOptions()}
                    {renderSummaryLengthOptions()}

                    <TouchableOpacity
                        style={styles.button}
                        onPress={handleSubmit}
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <ActivityIndicator color={COLORS.background} />
                        ) : (
                            <>
                                <Ionicons
                                    name="document-text"
                                    size={20}
                                    color={COLORS.background}
                                />
                                <Text style={styles.buttonText}>
                                    Generate Summary
                                </Text>
                            </>
                        )}
                    </TouchableOpacity>
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    scrollContent: {
        flexGrow: 1,
        padding: SPACING.lg,
    },
    header: {
        marginTop: SPACING.xl,
        marginBottom: SPACING.xl,
        alignItems: "center",
    },
    title: {
        fontSize: FONT_SIZES.xxxl,
        fontWeight: "bold",
        color: COLORS.primary,
        marginBottom: SPACING.xs,
    },
    subtitle: {
        fontSize: FONT_SIZES.md,
        color: COLORS.textSecondary,
        textAlign: "center",
    },
    inputContainer: {
        marginBottom: SPACING.lg,
    },
    inputRow: {
        flexDirection: "row",
        alignItems: "center",
    },
    input: {
        flex: 1,
        height: 50,
        borderWidth: 1,
        borderColor: COLORS.border,
        borderRadius: 8,
        paddingHorizontal: SPACING.md,
        fontSize: FONT_SIZES.md,
        backgroundColor: COLORS.surface,
    },
    inputError: {
        borderColor: COLORS.error,
    },
    pasteButton: {
        padding: SPACING.sm,
        marginLeft: SPACING.sm,
        borderRadius: 8,
        backgroundColor: COLORS.surface,
        borderWidth: 1,
        borderColor: COLORS.border,
        height: 50,
        width: 50,
        justifyContent: "center",
        alignItems: "center",
    },
    errorText: {
        color: COLORS.error,
        fontSize: FONT_SIZES.sm,
        marginTop: SPACING.xs,
    },
    optionsContainer: {
        marginBottom: SPACING.lg,
    },
    optionsLabel: {
        fontSize: FONT_SIZES.md,
        fontWeight: "500",
        marginBottom: SPACING.sm,
        color: COLORS.text,
    },
    optionsButtonGroup: {
        flexDirection: "row",
        flexWrap: "wrap",
    },
    optionButton: {
        paddingVertical: SPACING.sm,
        paddingHorizontal: SPACING.md,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: COLORS.border,
        marginRight: SPACING.sm,
        marginBottom: SPACING.sm,
        backgroundColor: COLORS.surface,
    },
    optionButtonSelected: {
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
    },
    optionButtonText: {
        fontSize: FONT_SIZES.sm,
        color: COLORS.text,
    },
    optionButtonTextSelected: {
        color: COLORS.background,
    },
    button: {
        backgroundColor: COLORS.primary,
        borderRadius: 8,
        paddingVertical: SPACING.md,
        flexDirection: "row",
        justifyContent: "center",
        alignItems: "center",
        marginTop: SPACING.md,
    },
    buttonText: {
        color: COLORS.background,
        fontSize: FONT_SIZES.md,
        fontWeight: "600",
        marginLeft: SPACING.sm,
    },
});

export default HomeScreen;
