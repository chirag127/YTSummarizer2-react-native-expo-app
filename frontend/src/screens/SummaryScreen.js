import React, { useState, useEffect, useRef } from "react";
import {
    StyleSheet,
    View,
    Text,
    Image,
    ScrollView,
    TouchableOpacity,
    ActivityIndicator,
    Alert,
    Share,
    Modal,
    Platform,
    FlatList,
    SafeAreaView,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Markdown from "react-native-markdown-display";
import { useTimeZone } from "../context/TimeZoneContext";

// Import components, services, and utilities
import {
    updateSummary,
    regenerateSummary,
    toggleStarSummary,
    getVideoSummaries,
    generateSummary,
} from "../services/api";
import {
    speakText,
    stopSpeaking,
    isSpeaking,
    setSpeechCallbacks,
    clearSpeechCallbacks,
    processTextForSpeech,
} from "../services/tts";
import {
    formatDate,
    truncateText,
    copyToClipboard,
    openUrl,
    formatSummaryType,
    formatSummaryLength,
    parseMarkdownToPlainText,
} from "../utils";
import {
    COLORS,
    SPACING,
    FONT_SIZES,
    SUMMARY_TYPES,
    SUMMARY_LENGTHS,
    SHADOWS,
} from "../constants";

const SummaryScreen = ({ route, navigation }) => {
    // Get summary from route params
    const { summary } = route.params || {};

    // Get time zone context
    const { getCurrentTimeZone, formatDateWithTimeZone } = useTimeZone();

    // State
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [editModalVisible, setEditModalVisible] = useState(false);
    const [showPlainText, setShowPlainText] = useState(false);
    const [selectedType, setSelectedType] = useState(
        summary?.summary_type || SUMMARY_TYPES[0].id
    );
    const [selectedLength, setSelectedLength] = useState(
        summary?.summary_length || SUMMARY_LENGTHS[1].id
    );
    const [otherSummaries, setOtherSummaries] = useState([]);
    const [loadingOtherSummaries, setLoadingOtherSummaries] = useState(false);
    const [showOtherSummaries, setShowOtherSummaries] = useState(false);
    const [generationStartTime, setGenerationStartTime] = useState(null);
    const [elapsedTime, setElapsedTime] = useState(0);
    const timerRef = useRef(null);

    // Handle generation time tracking
    useEffect(() => {
        if (isLoading && generationStartTime) {
            timerRef.current = setInterval(() => {
                setElapsedTime(
                    Math.floor((Date.now() - generationStartTime) / 1000)
                );
            }, 1000);
        }
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, [isLoading, generationStartTime]);

    // TTS highlighting state
    const [currentWord, setCurrentWord] = useState(null);
    const [currentSentence, setCurrentSentence] = useState(0); // Always start from the first sentence
    const [processedText, setProcessedText] = useState(null);

    // Refs
    const scrollViewRef = useRef(null);
    const sentenceRefs = useRef({});

    // Handle cancel regeneration
    let handleCancel = () => {
        // Immediately set loading to false to prevent further UI updates
        setIsLoading(false);
        setGenerationStartTime(null);
        setElapsedTime(0);
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
    };

    // Set navigation title
    useEffect(() => {
        navigation.setOptions({
            title: truncateText(summary?.video_title || "Summary", 30),
        });
    }, [navigation, summary]);

    // Fetch other summaries for the same video
    useEffect(() => {
        const fetchOtherSummaries = async () => {
            if (!summary?.video_url) return;

            setLoadingOtherSummaries(true);
            try {
                const response = await getVideoSummaries(summary.video_url);
                // Filter out the current summary
                const filteredSummaries = response.summaries.filter(
                    (s) => s.id !== summary.id
                );
                setOtherSummaries(filteredSummaries);
            } catch (error) {
                console.error("Error fetching other summaries:", error);
            } finally {
                setLoadingOtherSummaries(false);
            }
        };

        fetchOtherSummaries();
    }, [summary?.video_url, summary?.id]);

    // Check if TTS is playing
    useEffect(() => {
        const checkSpeakingStatus = async () => {
            const speaking = await isSpeaking();
            setIsPlaying(speaking);
        };

        const interval = setInterval(checkSpeakingStatus, 1000);
        return () => clearInterval(interval);
    }, []);

    // Process text for TTS when summary changes
    useEffect(() => {
        if (summary?.summary_text) {
            // Parse markdown to plain text for TTS processing
            const plainText = parseMarkdownToPlainText(summary.summary_text);
            const processed = processTextForSpeech(plainText);
            setProcessedText(processed);

            // Reset to the beginning of the summary when summary changes
            setCurrentSentence(0);
            setCurrentWord(null);
        }
    }, [summary]);

    // Setup speech callbacks
    useEffect(() => {
        setSpeechCallbacks({
            onBoundary: (event) => {
                // Update the current word with the information from the event
                setCurrentWord({
                    word: event.word,
                    sentenceIndex: event.sentenceIndex,
                    wordIndex: event.wordIndex,
                });
                setCurrentSentence(event.sentenceIndex);
            },
            onStart: (sentenceIndex) => {
                // When starting from the Read Aloud button, we want to start from the beginning
                // The sentenceIndex parameter will be 0 when starting from the beginning
                setCurrentSentence(sentenceIndex);
                // Reset current word when starting speech
                setCurrentWord(null);
            },
            onDone: () => {
                setCurrentWord(null);
                setIsPlaying(false);
            },
            onStopped: () => {
                setCurrentWord(null);
                setIsPlaying(false);
            },
        });

        // Clean up when component unmounts
        return () => {
            stopSpeaking();
            clearSpeechCallbacks();
        };
    }, []);

    // Scroll to the current word being spoken
    useEffect(() => {
        if (
            currentWord &&
            scrollViewRef.current &&
            sentenceRefs.current[currentWord.sentenceIndex]
        ) {
            // Get the sentence ref and measure its position
            sentenceRefs.current[currentWord.sentenceIndex].measureLayout(
                scrollViewRef.current,
                (_, y) => {
                    // Scroll to the position
                    scrollViewRef.current.scrollTo({
                        y: y,
                        animated: true,
                    });
                },
                () => console.log("Measurement failed")
            );
        }
    }, [currentWord]);

    // Handle play/pause
    const handlePlayPause = async () => {
        if (isPlaying) {
            await stopSpeaking();
            setIsPlaying(false);
        } else {
            // Use plain text for speech
            const plainText = parseMarkdownToPlainText(summary.summary_text);
            // Always start from the beginning (sentence index 0) when pressing Read Aloud
            setCurrentSentence(0); // Reset to first sentence
            const success = await speakText(plainText, 0); // Always start from the beginning
            setIsPlaying(success);
        }
    };

    // Handle next sentence
    const handleNextSentence = async () => {
        if (
            processedText &&
            currentSentence < processedText.sentences.length - 1
        ) {
            const nextSentence = currentSentence + 1;
            await stopSpeaking();
            // Use plain text for speech
            const plainText = parseMarkdownToPlainText(summary.summary_text);
            const success = await speakText(plainText, nextSentence);
            setIsPlaying(success);
        }
    };

    // Handle previous sentence
    const handlePrevSentence = async () => {
        if (processedText && currentSentence > 0) {
            const prevSentence = currentSentence - 1;
            await stopSpeaking();
            // Use plain text for speech
            const plainText = parseMarkdownToPlainText(summary.summary_text);
            const success = await speakText(plainText, prevSentence);
            setIsPlaying(success);
        }
    };

    // Handle share
    const handleShare = async () => {
        try {
            await Share.share({
                message: `Summary for "${summary.video_title}":\n\n${summary.summary_text}\n\nOriginal Video: ${summary.video_url}`,
            });
        } catch (error) {
            console.error("Error sharing summary:", error);
            Alert.alert("Error", "Failed to share summary.");
        }
    };

    // Handle copy
    const handleCopy = async () => {
        const success = await copyToClipboard(summary.summary_text);
        if (success) {
            Alert.alert("Success", "Summary copied to clipboard.");
        } else {
            Alert.alert("Error", "Failed to copy summary to clipboard.");
        }
    };

    // Handle open original video
    const handleOpenVideo = async () => {
        await openUrl(summary.video_url);
    };

    // Handle edit
    const handleEdit = () => {
        setSelectedType(summary.summary_type);
        setSelectedLength(summary.summary_length);
        setEditModalVisible(true);
    };

    // Handle navigation to another summary
    const handleNavigateToSummary = (otherSummary) => {
        // Stop TTS if playing
        if (isPlaying) {
            stopSpeaking();
            setIsPlaying(false);
        }

        // Navigate to the selected summary
        navigation.setParams({ summary: otherSummary });
    };

    // Handle star toggle
    const handleToggleStar = async () => {
        try {
            const newStarredStatus = !summary.is_starred;
            const updatedSummary = await toggleStarSummary(
                summary.id,
                newStarredStatus
            );

            // Update the route params with the updated summary
            navigation.setParams({ summary: updatedSummary });
        } catch (error) {
            console.error("Error toggling star status:", error);
            Alert.alert(
                "Error",
                "Failed to update star status. Please try again."
            );
        }
    };

    // Handle save edit - creates a new summary with the selected type and length
    const handleSaveEdit = async () => {
        if (
            selectedType === summary.summary_type &&
            selectedLength === summary.summary_length
        ) {
            setEditModalVisible(false);
            return;
        }

        // First, check if a summary with the selected type and length already exists
        try {
            // Fetch the latest summaries for this video to ensure we have the most up-to-date list
            const response = await getVideoSummaries(summary.video_url);
            const allSummaries = response.summaries;

            // Look for an existing summary with the same type and length
            const existingSummary = allSummaries.find(
                (s) =>
                    s.summary_type === selectedType &&
                    s.summary_length === selectedLength
            );

            if (existingSummary) {
                // A summary with these parameters already exists
                setEditModalVisible(false);

                // Silently navigate to the existing summary without showing an alert
                handleNavigateToSummary(existingSummary);
                return;
            }
        } catch (error) {
            console.error("Error checking for existing summaries:", error);
            // Continue with summary generation even if the check fails
        }

        // Create an abort controller for cancellation
        const abortController = new AbortController();

        // Use a local variable for accurate time tracking
        const startTime = Date.now();
        setIsLoading(true);
        setGenerationStartTime(startTime); // Update state for UI timer

        // Store the abort controller in a ref for the cancel button
        const cancelRef = {
            abort: () => {
                abortController.abort();
                setIsLoading(false);
                setGenerationStartTime(null);
                setElapsedTime(0);
                if (timerRef.current) {
                    clearInterval(timerRef.current);
                    timerRef.current = null;
                }
            },
        };

        // Update the cancel button handler
        const originalCancelHandler = handleCancel;
        handleCancel = () => {
            cancelRef.abort();
            originalCancelHandler();
        };

        try {
            // Use generateSummary to create a new summary instead of updating the existing one
            const newSummary = await generateSummary(
                summary.video_url,
                selectedType,
                selectedLength,
                abortController.signal
            );

            // If loading was cancelled, don't update UI
            if (!isLoading) return;

            // Calculate the elapsed time using the local variable for accuracy
            const timeTaken = Math.floor((Date.now() - startTime) / 1000);
            newSummary.timeTaken = timeTaken > 0 ? timeTaken : 1; // Ensure at least 1 second is shown

            // Refresh other summaries list
            const response = await getVideoSummaries(summary.video_url);
            const filteredSummaries = response.summaries.filter(
                (s) => s.id !== newSummary.id
            );
            setOtherSummaries(filteredSummaries);

            // Automatically show and highlight the other summaries section
            if (filteredSummaries.length > 0) {
                setShowOtherSummaries(true);
            }

            // Update the route params with the new summary
            navigation.setParams({ summary: newSummary });
        } catch (error) {
            // Don't show error if it was cancelled
            if (error.name === "AbortError" || !isLoading) return;

            console.error("Error creating new summary:", error);

            Alert.alert(
                "Error",
                error.response?.data?.detail || "Failed to create new summary."
            );
        } finally {
            // Restore original cancel handler
            handleCancel = originalCancelHandler;

            // Close the modal if we're still in loading state (i.e., the generation wasn't cancelled by user)
            if (isLoading) {
                setEditModalVisible(false);
            }

            setIsLoading(false);
            setGenerationStartTime(null);
            setElapsedTime(0);
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    };

    // Handle regenerate summary - regenerates the summary with the same parameters
    const handleRegenerateSummary = async () => {
        // Create an abort controller for cancellation
        const abortController = new AbortController();

        // Use a local variable for accurate time tracking
        const startTime = Date.now();
        setIsLoading(true);
        setGenerationStartTime(startTime); // Update state for UI timer

        // Store the abort controller in a ref for the cancel button
        const cancelRef = {
            abort: () => {
                abortController.abort();
                setIsLoading(false);
                setGenerationStartTime(null);
                setElapsedTime(0);
                if (timerRef.current) {
                    clearInterval(timerRef.current);
                    timerRef.current = null;
                }
            },
        };

        // Update the cancel button handler
        const originalCancelHandler = handleCancel;
        handleCancel = () => {
            cancelRef.abort();
            originalCancelHandler();
        };

        try {
            const newSummary = await regenerateSummary(summary.id);

            // If loading was cancelled, don't update UI
            if (!isLoading) return;

            // Calculate the elapsed time using the local variable for accuracy
            const timeTaken = Math.floor((Date.now() - startTime) / 1000);
            newSummary.timeTaken = timeTaken > 0 ? timeTaken : 1; // Ensure at least 1 second is shown

            // Update the route params with the new summary
            navigation.setParams({ summary: newSummary });

            // Refresh other summaries list
            const response = await getVideoSummaries(summary.video_url);
            const filteredSummaries = response.summaries.filter(
                (s) => s.id !== newSummary.id
            );
            setOtherSummaries(filteredSummaries);

            // Show success message
            Alert.alert(
                "Success",
                "Summary has been regenerated successfully."
            );
        } catch (error) {
            // Don't show error if it was cancelled
            if (error.name === "AbortError" || !isLoading) return;

            console.error("Error regenerating summary:", error);
            Alert.alert(
                "Error",
                error.response?.data?.detail ||
                    "Failed to regenerate summary. Please try again."
            );
        } finally {
            // Restore original cancel handler
            handleCancel = originalCancelHandler;

            setIsLoading(false);
            setGenerationStartTime(null);
            setElapsedTime(0);
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    };

    // Render edit modal
    const renderEditModal = () => {
        return (
            <Modal
                visible={editModalVisible}
                transparent={true}
                animationType="slide"
                onRequestClose={() => setEditModalVisible(false)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <Text style={styles.modalTitle}>
                            Create New Summary
                        </Text>

                        {isLoading ? (
                            <View style={styles.loadingContainer}>
                                <ActivityIndicator
                                    size="small"
                                    color={COLORS.primary}
                                />
                                <Text style={styles.loadingText}>
                                    Generating summary... {elapsedTime}s
                                </Text>
                                <TouchableOpacity
                                    style={styles.cancelButton}
                                    onPress={() => {
                                        handleCancel();
                                        // Close the modal when the user cancels the generation
                                        setEditModalVisible(false);
                                    }}
                                >
                                    <Ionicons
                                        name="close-circle"
                                        size={20}
                                        color={COLORS.error}
                                    />
                                    <Text style={styles.cancelButtonText}>
                                        Cancel
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        ) : (
                            <>
                                <Text style={styles.modalLabel}>
                                    Summary Type:
                                </Text>
                                <View style={styles.optionsButtonGroup}>
                                    {SUMMARY_TYPES.map((type) => (
                                        <TouchableOpacity
                                            key={type.id}
                                            style={[
                                                styles.optionButton,
                                                selectedType === type.id &&
                                                    styles.optionButtonSelected,
                                            ]}
                                            onPress={() =>
                                                setSelectedType(type.id)
                                            }
                                        >
                                            <Text
                                                style={[
                                                    styles.optionButtonText,
                                                    selectedType === type.id &&
                                                        styles.optionButtonTextSelected,
                                                ]}
                                            >
                                                {type.label}
                                            </Text>
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                <Text style={styles.modalLabel}>
                                    Summary Length:
                                </Text>
                                <View style={styles.optionsButtonGroup}>
                                    {SUMMARY_LENGTHS.map((length) => (
                                        <TouchableOpacity
                                            key={length.id}
                                            style={[
                                                styles.optionButton,
                                                selectedLength === length.id &&
                                                    styles.optionButtonSelected,
                                            ]}
                                            onPress={() =>
                                                setSelectedLength(length.id)
                                            }
                                        >
                                            <Text
                                                style={[
                                                    styles.optionButtonText,
                                                    selectedLength ===
                                                        length.id &&
                                                        styles.optionButtonTextSelected,
                                                ]}
                                            >
                                                {length.label}
                                            </Text>
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                <View style={styles.modalButtons}>
                                    <TouchableOpacity
                                        style={[
                                            styles.modalButton,
                                            styles.modalCancelButton,
                                        ]}
                                        onPress={() =>
                                            setEditModalVisible(false)
                                        }
                                    >
                                        <Text style={styles.modalButtonText}>
                                            Cancel
                                        </Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity
                                        style={[
                                            styles.modalButton,
                                            styles.modalSaveButton,
                                        ]}
                                        onPress={handleSaveEdit}
                                        disabled={isLoading}
                                    >
                                        <Text
                                            style={[
                                                styles.modalButtonText,
                                                styles.modalSaveButtonText,
                                            ]}
                                        >
                                            Create
                                        </Text>
                                    </TouchableOpacity>
                                </View>
                            </>
                        )}
                    </View>
                </View>
            </Modal>
        );
    };

    // If no summary, show error
    if (!summary) {
        return (
            <View style={styles.errorContainer}>
                <Text style={styles.errorText}>Summary not found.</Text>
            </View>
        );
    }

    // Render loading overlay for regeneration
    const renderLoadingOverlay = () => {
        if (!isLoading || editModalVisible) return null;

        return (
            <View style={styles.loadingOverlay}>
                <View style={styles.loadingCard}>
                    <ActivityIndicator size="large" color={COLORS.primary} />
                    <Text style={styles.loadingText}>
                        Regenerating summary... {elapsedTime}s
                    </Text>
                    <TouchableOpacity
                        style={styles.cancelButton}
                        onPress={handleCancel}
                    >
                        <Ionicons
                            name="close-circle"
                            size={20}
                            color={COLORS.error}
                        />
                        <Text style={styles.cancelButtonText}>Cancel</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    };

    return (
        <SafeAreaView style={styles.container}>
            {renderEditModal()}
            {renderLoadingOverlay()}

            <ScrollView
                ref={scrollViewRef}
                contentContainerStyle={styles.scrollContent}
            >
                {/* Video Info */}
                <View style={styles.videoInfoContainer}>
                    <Image
                        source={{
                            uri:
                                summary.video_thumbnail_url ||
                                "https://via.placeholder.com/480x360?text=No+Thumbnail",
                        }}
                        style={styles.thumbnail}
                        resizeMode="cover"
                    />
                    <Text style={styles.videoTitle}>{summary.video_title}</Text>
                    <TouchableOpacity
                        style={styles.videoLinkButton}
                        onPress={handleOpenVideo}
                    >
                        <Ionicons
                            name="logo-youtube"
                            size={16}
                            color={COLORS.error}
                        />
                        <Text style={styles.videoLinkText}>
                            Watch Original Video
                        </Text>
                    </TouchableOpacity>
                </View>

                {/* Summary Info */}
                <View style={styles.summaryInfoContainer}>
                    <View style={styles.summaryTypeContainer}>
                        <View style={styles.summaryTypeItem}>
                            <Text style={styles.summaryTypeLabel}>Type:</Text>
                            <Text style={styles.summaryTypeValue}>
                                {formatSummaryType(summary.summary_type)}
                            </Text>
                        </View>
                        <View style={styles.summaryTypeItem}>
                            <Text style={styles.summaryTypeLabel}>Length:</Text>
                            <Text style={styles.summaryTypeValue}>
                                {formatSummaryLength(summary.summary_length)}
                            </Text>
                        </View>
                        <View style={styles.summaryTypeItem}>
                            <Text style={styles.summaryTypeLabel}>
                                Created:
                            </Text>
                            <Text style={styles.summaryTypeValue}>
                                {formatDateWithTimeZone(summary.created_at)}
                            </Text>
                        </View>
                        {summary.timeTaken !== undefined && (
                            <View style={styles.summaryTypeItem}>
                                <Text style={styles.summaryTypeLabel}>
                                    Time Taken:
                                </Text>
                                <Text style={styles.summaryTypeValue}>
                                    {summary.timeTaken} seconds
                                </Text>
                            </View>
                        )}
                    </View>
                </View>

                {/* Summary Content */}
                <View style={styles.summaryContentContainer}>
                    <Text style={styles.summaryTitle}>Summary</Text>
                    {(isPlaying || showPlainText) && processedText ? (
                        <View>
                            {processedText.sentences.map((sentence, index) => (
                                <View
                                    key={`sentence-${index}`}
                                    ref={(ref) =>
                                        (sentenceRefs.current[index] = ref)
                                    }
                                    style={[
                                        styles.sentenceContainer,
                                        currentSentence === index &&
                                            styles.activeSentence,
                                    ]}
                                >
                                    {sentence
                                        .split(/\s+/)
                                        .map((word, wordIdx) => {
                                            // Check if this word should be highlighted
                                            // We need to make sure the word is not empty
                                            if (word.trim() === "") return null;

                                            const isHighlighted =
                                                !showPlainText &&
                                                currentWord &&
                                                currentWord.sentenceIndex ===
                                                    index &&
                                                currentWord.wordIndex ===
                                                    wordIdx;

                                            return (
                                                <Text
                                                    key={`word-${index}-${wordIdx}`}
                                                    style={[
                                                        styles.word,
                                                        isHighlighted &&
                                                            styles.highlightedWord,
                                                    ]}
                                                    selectable={true}
                                                >
                                                    {word}{" "}
                                                </Text>
                                            );
                                        })}
                                </View>
                            ))}
                        </View>
                    ) : (
                        <Markdown style={markdownStyles} selectable={true}>
                            {summary.summary_text}
                        </Markdown>
                    )}
                </View>

                {/* Other Summaries Section */}
                {otherSummaries.length > 0 && (
                    <View style={styles.otherSummariesContainer}>
                        <View style={styles.otherSummariesHeader}>
                            <Text style={styles.otherSummariesTitle}>
                                Other Summaries for This Video
                            </Text>
                            <TouchableOpacity
                                onPress={() =>
                                    setShowOtherSummaries(!showOtherSummaries)
                                }
                            >
                                <Ionicons
                                    name={
                                        showOtherSummaries
                                            ? "chevron-up"
                                            : "chevron-down"
                                    }
                                    size={24}
                                    color={COLORS.primary}
                                />
                            </TouchableOpacity>
                        </View>

                        {showOtherSummaries && (
                            <FlatList
                                data={otherSummaries}
                                keyExtractor={(item) => item.id}
                                renderItem={({ item }) => (
                                    <TouchableOpacity
                                        style={styles.otherSummaryItem}
                                        onPress={() =>
                                            handleNavigateToSummary(item)
                                        }
                                    >
                                        <View style={styles.otherSummaryInfo}>
                                            <View
                                                style={
                                                    styles.otherSummaryBadges
                                                }
                                            >
                                                <View
                                                    style={[
                                                        styles.badge,
                                                        styles.typeBadge,
                                                    ]}
                                                >
                                                    <Text
                                                        style={styles.badgeText}
                                                    >
                                                        {item.summary_type}
                                                    </Text>
                                                </View>
                                                <View
                                                    style={[
                                                        styles.badge,
                                                        styles.lengthBadge,
                                                    ]}
                                                >
                                                    <Text
                                                        style={styles.badgeText}
                                                    >
                                                        {item.summary_length}
                                                    </Text>
                                                </View>
                                            </View>
                                            <Text
                                                style={styles.otherSummaryDate}
                                            >
                                                {formatDateWithTimeZone(
                                                    item.created_at
                                                )}
                                            </Text>
                                        </View>
                                        <Ionicons
                                            name="chevron-forward"
                                            size={20}
                                            color={COLORS.textSecondary}
                                        />
                                    </TouchableOpacity>
                                )}
                                style={styles.otherSummariesList}
                                scrollEnabled={false}
                                nestedScrollEnabled={true}
                            />
                        )}
                    </View>
                )}
            </ScrollView>

            {/* TTS Navigation Buttons */}
            {isPlaying && !showPlainText && (
                <View style={styles.ttsNavigationContainer}>
                    <TouchableOpacity
                        style={[
                            styles.ttsNavButton,
                            currentSentence === 0 &&
                                styles.ttsNavButtonDisabled,
                        ]}
                        onPress={handlePrevSentence}
                        disabled={currentSentence === 0}
                    >
                        <Ionicons
                            name="arrow-back"
                            size={24}
                            color={
                                currentSentence === 0
                                    ? COLORS.textSecondary
                                    : COLORS.primary
                            }
                        />
                        <Text style={styles.ttsNavButtonText}>Previous</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={[
                            styles.ttsNavButton,
                            processedText &&
                                currentSentence ===
                                    processedText.sentences.length - 1 &&
                                styles.ttsNavButtonDisabled,
                        ]}
                        onPress={handleNextSentence}
                        disabled={
                            processedText &&
                            currentSentence ===
                                processedText.sentences.length - 1
                        }
                    >
                        <Ionicons
                            name="arrow-forward"
                            size={24}
                            color={
                                processedText &&
                                currentSentence ===
                                    processedText.sentences.length - 1
                                    ? COLORS.textSecondary
                                    : COLORS.primary
                            }
                        />
                        <Text style={styles.ttsNavButtonText}>Next</Text>
                    </TouchableOpacity>
                </View>
            )}

            {/* Action Buttons */}
            <View style={styles.actionButtonsContainer}>
                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handlePlayPause}
                >
                    <Ionicons
                        name={isPlaying ? "pause-circle" : "play-circle"}
                        size={24}
                        color={COLORS.primary}
                    />
                    <Text style={styles.actionButtonText}>
                        {isPlaying ? "Pause" : "Read Aloud"}
                    </Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => {
                        // If TTS is playing, stop it
                        if (isPlaying) {
                            stopSpeaking();
                            setIsPlaying(false);
                        }
                        setShowPlainText(!showPlainText);
                    }}
                >
                    <Ionicons
                        name={showPlainText ? "document" : "document-text"}
                        size={24}
                        color={COLORS.primary}
                    />
                    <Text style={styles.actionButtonText}>
                        {showPlainText ? "Show Markdown" : "Show Text"}
                    </Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handleToggleStar}
                >
                    <Ionicons
                        name={summary.is_starred ? "star" : "star-outline"}
                        size={24}
                        color={
                            summary.is_starred ? COLORS.accent : COLORS.primary
                        }
                    />
                    <Text style={styles.actionButtonText}>
                        {summary.is_starred ? "Starred" : "Star"}
                    </Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handleShare}
                >
                    <Ionicons
                        name="share-social"
                        size={24}
                        color={COLORS.primary}
                    />
                    <Text style={styles.actionButtonText}>Share</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handleCopy}
                >
                    <Ionicons name="copy" size={24} color={COLORS.primary} />
                    <Text style={styles.actionButtonText}>Copy</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handleRegenerateSummary}
                    disabled={isLoading}
                >
                    <Ionicons
                        name="refresh-outline"
                        size={24}
                        color={isLoading ? COLORS.disabled : COLORS.primary}
                    />
                    <Text
                        style={[
                            styles.actionButtonText,
                            isLoading && { color: COLORS.disabled },
                        ]}
                    >
                        Regenerate
                    </Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={handleEdit}
                >
                    <Ionicons
                        name="add-circle-outline"
                        size={24}
                        color={COLORS.primary}
                    />
                    <Text style={styles.actionButtonText}>New Type</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => navigation.navigate("QA", { summary })}
                >
                    <Ionicons
                        name="chatbubble-outline"
                        size={24}
                        color={COLORS.primary}
                    />
                    <Text style={styles.actionButtonText}>Ask AI</Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    // Loading overlay styles
    loadingOverlay: {
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 1000,
    },
    loadingCard: {
        backgroundColor: COLORS.background,
        borderRadius: 8,
        padding: SPACING.lg,
        width: "80%",
        maxWidth: 300,
        alignItems: "center",
        ...SHADOWS.medium,
    },

    // TTS Highlighting styles
    sentenceContainer: {
        flexDirection: "row",
        flexWrap: "wrap",
        marginBottom: SPACING.md,
    },
    activeSentence: {
        backgroundColor: "rgba(0, 123, 255, 0.05)",
        borderRadius: 4,
        padding: SPACING.xs,
    },
    word: {
        fontSize: FONT_SIZES.md,
        color: COLORS.text,
        lineHeight: 22,
    },
    highlightedWord: {
        backgroundColor: "rgba(0, 123, 255, 0.4)",
        borderRadius: 4,
        fontWeight: "600",
        color: COLORS.primary,
    },
    ttsNavigationContainer: {
        flexDirection: "row",
        justifyContent: "space-between",
        paddingHorizontal: SPACING.lg,
        paddingVertical: SPACING.sm,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
        backgroundColor: COLORS.background,
    },
    ttsNavButton: {
        flexDirection: "row",
        alignItems: "center",
        padding: SPACING.sm,
    },
    ttsNavButtonDisabled: {
        opacity: 0.5,
    },
    ttsNavButtonText: {
        fontSize: FONT_SIZES.sm,
        color: COLORS.text,
        marginLeft: SPACING.xs,
    },
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    scrollContent: {
        padding: SPACING.md,
    },
    errorContainer: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        padding: SPACING.lg,
    },
    errorText: {
        fontSize: FONT_SIZES.lg,
        color: COLORS.error,
        textAlign: "center",
    },
    videoInfoContainer: {
        marginBottom: SPACING.lg,
        alignItems: "center",
    },
    thumbnail: {
        width: "100%",
        height: 200,
        borderRadius: 8,
        marginBottom: SPACING.md,
    },
    videoTitle: {
        fontSize: FONT_SIZES.lg,
        fontWeight: "bold",
        color: COLORS.text,
        textAlign: "center",
        marginBottom: SPACING.sm,
    },
    videoLinkButton: {
        flexDirection: "row",
        alignItems: "center",
        padding: SPACING.sm,
    },
    videoLinkText: {
        color: COLORS.primary,
        marginLeft: SPACING.xs,
        fontSize: FONT_SIZES.sm,
    },
    summaryInfoContainer: {
        marginBottom: SPACING.lg,
    },
    summaryTypeContainer: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "space-between",
        backgroundColor: COLORS.surface,
        borderRadius: 8,
        padding: SPACING.md,
    },
    summaryTypeItem: {
        marginBottom: SPACING.sm,
        minWidth: "30%",
    },
    summaryTypeLabel: {
        fontSize: FONT_SIZES.sm,
        color: COLORS.textSecondary,
        marginBottom: 2,
    },
    summaryTypeValue: {
        fontSize: FONT_SIZES.md,
        color: COLORS.text,
        fontWeight: "500",
    },
    summaryContentContainer: {
        backgroundColor: COLORS.surface,
        borderRadius: 8,
        padding: SPACING.md,
        marginBottom: SPACING.lg,
    },
    summaryTitle: {
        fontSize: FONT_SIZES.lg,
        fontWeight: "bold",
        color: COLORS.text,
        marginBottom: SPACING.md,
    },
    actionButtonsContainer: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "space-around",
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
        paddingVertical: SPACING.md,
        backgroundColor: COLORS.background,
    },
    actionButton: {
        alignItems: "center",
        justifyContent: "center",
        padding: SPACING.sm,
        minWidth: 70,
        marginHorizontal: 2,
        marginVertical: 4,
    },
    actionButtonText: {
        fontSize: FONT_SIZES.xs,
        color: COLORS.text,
        marginTop: 4,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
    },
    modalContent: {
        backgroundColor: COLORS.background,
        borderRadius: 8,
        padding: SPACING.lg,
        width: "80%",
        maxWidth: 400,
    },
    modalTitle: {
        fontSize: FONT_SIZES.xl,
        fontWeight: "bold",
        color: COLORS.text,
        marginBottom: SPACING.lg,
        textAlign: "center",
    },
    modalLabel: {
        fontSize: FONT_SIZES.md,
        fontWeight: "500",
        color: COLORS.text,
        marginBottom: SPACING.sm,
    },
    optionsButtonGroup: {
        flexDirection: "row",
        flexWrap: "wrap",
        marginBottom: SPACING.lg,
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
    modalButtons: {
        flexDirection: "row",
        justifyContent: "space-between",
        marginTop: SPACING.md,
    },
    modalButton: {
        flex: 1,
        paddingVertical: SPACING.md,
        borderRadius: 8,
        alignItems: "center",
        justifyContent: "center",
    },
    modalCancelButton: {
        backgroundColor: COLORS.surface,
        marginRight: SPACING.sm,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    modalSaveButton: {
        backgroundColor: COLORS.primary,
        marginLeft: SPACING.sm,
    },
    modalButtonText: {
        fontSize: FONT_SIZES.md,
        fontWeight: "600",
        color: COLORS.text,
    },
    modalSaveButtonText: {
        color: COLORS.background,
    },
    loadingContainer: {
        alignItems: "center",
        justifyContent: "center",
        marginBottom: SPACING.md,
    },
    loadingText: {
        fontSize: FONT_SIZES.sm,
        color: COLORS.textSecondary,
        marginTop: SPACING.sm,
    },
    cancelButton: {
        flexDirection: "row",
        alignItems: "center",
        marginTop: SPACING.md,
    },
    cancelButtonText: {
        fontSize: FONT_SIZES.sm,
        color: COLORS.error,
        marginLeft: SPACING.xs,
    },
    // Other summaries styles
    otherSummariesContainer: {
        backgroundColor: COLORS.surface,
        borderRadius: 8,
        padding: SPACING.md,
        marginBottom: SPACING.lg,
    },
    otherSummariesHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: SPACING.md,
    },
    otherSummariesTitle: {
        fontSize: FONT_SIZES.lg,
        fontWeight: "bold",
        color: COLORS.text,
    },
    otherSummariesList: {
        marginTop: SPACING.sm,
    },
    otherSummaryItem: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingVertical: SPACING.sm,
        paddingHorizontal: SPACING.sm,
        borderBottomWidth: 1,
        borderBottomColor: COLORS.border,
    },
    otherSummaryInfo: {
        flex: 1,
    },
    otherSummaryBadges: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 4,
        gap: 8,
    },
    badge: {
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 12,
        alignItems: "center",
        justifyContent: "center",
    },
    typeBadge: {
        backgroundColor: COLORS.primary,
    },
    lengthBadge: {
        backgroundColor: COLORS.secondary,
    },
    badgeText: {
        fontSize: FONT_SIZES.xs,
        color: "white",
        fontWeight: "500",
    },
    otherSummaryDate: {
        fontSize: FONT_SIZES.xs,
        color: COLORS.textSecondary,
    },
});

const markdownStyles = {
    body: {
        color: COLORS.text,
        fontSize: FONT_SIZES.md,
    },
    heading1: {
        fontSize: FONT_SIZES.xxl,
        fontWeight: "bold",
        color: COLORS.text,
        marginTop: SPACING.lg,
        marginBottom: SPACING.md,
    },
    heading2: {
        fontSize: FONT_SIZES.xl,
        fontWeight: "bold",
        color: COLORS.text,
        marginTop: SPACING.lg,
        marginBottom: SPACING.md,
    },
    heading3: {
        fontSize: FONT_SIZES.lg,
        fontWeight: "bold",
        color: COLORS.text,
        marginTop: SPACING.md,
        marginBottom: SPACING.sm,
    },
    paragraph: {
        marginBottom: SPACING.md,
        lineHeight: 22,
    },
    list_item: {
        marginBottom: SPACING.sm,
    },
    bullet_list: {
        marginBottom: SPACING.md,
    },
    ordered_list: {
        marginBottom: SPACING.md,
    },
    blockquote: {
        borderLeftWidth: 4,
        borderLeftColor: COLORS.primary,
        paddingLeft: SPACING.md,
        marginLeft: SPACING.sm,
        marginBottom: SPACING.md,
        opacity: 0.8,
    },
    code_block: {
        backgroundColor: "#f5f5f5",
        padding: SPACING.md,
        borderRadius: 4,
        fontFamily: Platform.OS === "ios" ? "Courier" : "monospace",
        marginBottom: SPACING.md,
    },
    code_inline: {
        backgroundColor: "#f5f5f5",
        fontFamily: Platform.OS === "ios" ? "Courier" : "monospace",
        padding: 4,
        borderRadius: 2,
    },
    link: {
        color: COLORS.primary,
        textDecorationLine: "underline",
    },
    strong: {
        fontWeight: "bold",
    },
    em: {
        fontStyle: "italic",
    },
};

export default SummaryScreen;
