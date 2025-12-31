"""Unit tests for Heart-JEPA models."""

import pytest
import torch


class TestViTEncoder:
    """Tests for ViT encoder."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample spectrogram input."""
        return torch.randn(2, 1, 224, 224)

    def test_encoder_output_shape(self, sample_input):
        """Test encoder output shape."""
        from heart_jepa.models.encoder import ViTEncoder

        encoder = ViTEncoder(pretrained=False)
        output = encoder(sample_input)

        # Should output (B, embed_dim)
        assert output.shape == (2, encoder.embed_dim)

    def test_encoder_forward_features(self, sample_input):
        """Test encoder forward_features method."""
        from heart_jepa.models.encoder import ViTEncoder

        encoder = ViTEncoder(pretrained=False)
        features = encoder.forward_features(sample_input)

        # Should output (B, N+1, embed_dim) where N=196 for 14x14 patches
        B = sample_input.shape[0]
        assert features.shape[0] == B
        assert features.shape[1] == 197  # 14*14 + 1 (CLS token)
        assert features.shape[2] == encoder.embed_dim

    def test_encoder_single_channel(self):
        """Test encoder with single channel input."""
        from heart_jepa.models.encoder import ViTEncoder

        encoder = ViTEncoder(pretrained=False, in_chans=1)
        x = torch.randn(1, 1, 224, 224)
        output = encoder(x)

        assert output.shape == (1, encoder.embed_dim)


class TestHeartJEPAEncoder:
    """Tests for Heart-JEPA encoder with projector."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample spectrogram input."""
        return torch.randn(2, 1, 224, 224)

    @pytest.fixture
    def multi_view_input(self):
        """Generate multi-view input."""
        return torch.randn(2, 4, 1, 224, 224)  # B=2, V=4 views

    def test_encoder_single_view_output(self, sample_input):
        """Test encoder output for single view input."""
        from heart_jepa.models.encoder import HeartJEPAEncoder

        encoder = HeartJEPAEncoder(pretrained=False)
        features, projections = encoder(sample_input)

        # Features: (B, embed_dim)
        assert features.shape == (2, encoder.embed_dim)
        # Projections: (B, 1, proj_dim)
        assert projections.shape == (2, 1, encoder.proj_dim)

    def test_encoder_multi_view_output(self, multi_view_input):
        """Test encoder output for multi-view input."""
        from heart_jepa.models.encoder import HeartJEPAEncoder

        encoder = HeartJEPAEncoder(pretrained=False)
        features, projections = encoder(multi_view_input)

        B, V = 2, 4
        # Features: (B*V, embed_dim)
        assert features.shape == (B * V, encoder.embed_dim)
        # Projections: (B, V, proj_dim)
        assert projections.shape == (B, V, encoder.proj_dim)

    def test_encoder_return_all_tokens(self, sample_input):
        """Test encoder with return_all_tokens=True."""
        from heart_jepa.models.encoder import HeartJEPAEncoder

        encoder = HeartJEPAEncoder(pretrained=False)
        features, projections = encoder(sample_input, return_all_tokens=True)

        # Features: (B, N+1, embed_dim)
        assert features.shape == (2, 197, encoder.embed_dim)


class TestSegmentationHead:
    """Tests for segmentation head."""

    @pytest.fixture
    def sample_patch_tokens(self):
        """Generate sample patch tokens."""
        # 196 patches from 14x14 grid
        return torch.randn(2, 196, 768)

    def test_segmentation_head_output_shape(self, sample_patch_tokens):
        """Test segmentation head output shape."""
        from heart_jepa.models.heads import SegmentationHead

        head = SegmentationHead(
            embed_dim=768,
            num_classes=7,
            output_frames=224,
        )
        logits = head(sample_patch_tokens)

        # Output: (B, num_classes, output_frames)
        assert logits.shape == (2, 7, 224)

    def test_segmentation_head_different_frames(self, sample_patch_tokens):
        """Test segmentation head with different output frames."""
        from heart_jepa.models.heads import SegmentationHead

        for output_frames in [128, 224, 512]:
            head = SegmentationHead(
                embed_dim=768,
                num_classes=7,
                output_frames=output_frames,
            )
            logits = head(sample_patch_tokens)
            assert logits.shape == (2, 7, output_frames)


class TestClassificationHead:
    """Tests for classification head."""

    @pytest.fixture
    def sample_cls_token(self):
        """Generate sample CLS token."""
        return torch.randn(2, 768)

    def test_classification_head_output_shape(self, sample_cls_token):
        """Test classification head output shape."""
        from heart_jepa.models.heads import ClassificationHead

        head = ClassificationHead(
            embed_dim=768,
            num_classes=3,
        )
        logits = head(sample_cls_token)

        # Output: (B, num_classes)
        assert logits.shape == (2, 3)

    def test_classification_head_different_classes(self, sample_cls_token):
        """Test classification head with different number of classes."""
        from heart_jepa.models.heads import ClassificationHead

        for num_classes in [2, 3, 5]:
            head = ClassificationHead(
                embed_dim=768,
                num_classes=num_classes,
            )
            logits = head(sample_cls_token)
            assert logits.shape == (2, num_classes)


class TestHeartJEPA:
    """Tests for full Heart-JEPA model."""

    @pytest.fixture
    def sample_input(self):
        """Generate sample spectrogram input."""
        return torch.randn(2, 1, 224, 224)

    @pytest.fixture
    def multi_view_input(self):
        """Generate multi-view input."""
        return torch.randn(2, 4, 1, 224, 224)

    def test_heart_jepa_forward_single_view(self, sample_input):
        """Test Heart-JEPA forward pass with single view."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)
        proj, seg_logits, cls_logits = model(sample_input)

        # Projections: (B, 1, proj_dim)
        assert proj.shape == (2, 1, 256)
        # Segmentation: (B, 7, 224)
        assert seg_logits.shape == (2, 7, 224)
        # Classification: (B, 3)
        assert cls_logits.shape == (2, 3)

    def test_heart_jepa_forward_multi_view(self, multi_view_input):
        """Test Heart-JEPA forward pass with multi-view."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)
        proj, seg_logits, cls_logits = model(multi_view_input)

        # Projections: (B, V, proj_dim)
        assert proj.shape == (2, 4, 256)
        # Segmentation: (B, 7, 224) - uses first view
        assert seg_logits.shape == (2, 7, 224)
        # Classification: (B, 3)
        assert cls_logits.shape == (2, 3)

    def test_heart_jepa_return_features(self, sample_input):
        """Test Heart-JEPA with return_features=True."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)
        proj, seg_logits, cls_logits, features = model(sample_input, return_features=True)

        # Features should be returned
        assert features is not None
        assert features.shape[0] == 2  # batch size

    def test_heart_jepa_forward_segmentation(self, sample_input):
        """Test Heart-JEPA segmentation-only forward."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)
        seg_logits = model.forward_segmentation(sample_input)

        assert seg_logits.shape == (2, 7, 224)

    def test_heart_jepa_forward_classification(self, sample_input):
        """Test Heart-JEPA classification-only forward."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)
        cls_logits = model.forward_classification(sample_input)

        assert cls_logits.shape == (2, 3)

    def test_heart_jepa_freeze_encoder(self, sample_input):
        """Test encoder freezing."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)

        # Freeze encoder
        model.freeze_encoder()
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # Unfreeze encoder
        model.unfreeze_encoder()
        for param in model.encoder.parameters():
            assert param.requires_grad

    def test_heart_jepa_get_params(self):
        """Test parameter getter methods."""
        from heart_jepa.models import HeartJEPA

        model = HeartJEPA(pretrained=False)

        encoder_params = list(model.get_encoder_params())
        head_params = list(model.get_head_params())

        assert len(encoder_params) > 0
        assert len(head_params) > 0


class TestLinearProbe:
    """Tests for linear probe."""

    def test_linear_probe_output(self):
        """Test linear probe output shape."""
        from heart_jepa.models.heads import LinearProbe

        probe = LinearProbe(embed_dim=768, num_classes=2)
        x = torch.randn(4, 768)
        output = probe(x)

        assert output.shape == (4, 2)


class TestLosses:
    """Tests for loss functions using official LEJEPA."""

    def test_sigreg_loss_basic(self):
        """Test SIGReg loss computation using LEJEPA."""
        from heart_jepa.losses import sigreg_loss

        # Use smaller num_slices for faster testing
        embeddings = torch.randn(32, 256)
        loss = sigreg_loss(embeddings, num_slices=64)

        assert loss.shape == ()
        assert loss >= 0

    def test_sigreg_module(self):
        """Test SIGReg loss module using LEJEPA."""
        from heart_jepa.losses import SIGReg

        # Use smaller num_slices for faster testing
        sigreg = SIGReg(num_slices=64, weight=0.5)
        embeddings = torch.randn(32, 256)
        loss = sigreg(embeddings)

        assert loss.shape == ()

    def test_sigreg_multi_view(self):
        """Test SIGReg with multi-view embeddings using LEJEPA."""
        from heart_jepa.losses import SIGReg

        # Use smaller num_slices for faster testing
        sigreg = SIGReg(num_slices=64)
        embeddings = torch.randn(8, 4, 256)  # B=8, V=4, D=256
        loss = sigreg(embeddings)

        assert loss.shape == ()

    def test_sigreg_uses_lejepa(self):
        """Verify SIGReg uses official LEJEPA components."""
        from heart_jepa.losses import SIGReg
        from lejepa.multivariate import SlicingUnivariateTest

        sigreg = SIGReg(num_slices=64)

        # Check that it uses LEJEPA's SlicingUnivariateTest
        assert isinstance(sigreg.test, SlicingUnivariateTest)

    def test_invariance_loss(self):
        """Test invariance loss."""
        from heart_jepa.losses import invariance_loss

        embeddings = torch.randn(8, 4, 256)  # B=8, V=4, D=256
        loss = invariance_loss(embeddings)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_segmentation_loss(self):
        """Test segmentation loss."""
        from heart_jepa.losses import SegmentationLoss

        loss_fn = SegmentationLoss(num_classes=7)

        logits = torch.randn(4, 7, 224)  # B=4, C=7, T=224
        targets = torch.randint(0, 7, (4, 224))  # B=4, T=224

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
